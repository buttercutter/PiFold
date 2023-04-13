import torch
import torch.nn.functional as F
import numpy as np
from collections.abc import Mapping, Sequence

"""
Notation:
    - B: Batch size
    - N: Number of nodes
    - E: Number of edges
    - K: K-nearest neighbors
    - C: Chain atoms per residue
    - R: Rotation basis vectors
    - D: Dimensions
"""


# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def cal_dihedral(X, eps=1e-7):
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:,:-2,:] # CA-N, C-CA, N-C,...
    u_1 = U[:,1:-1,:] # C-CA, N-C, CA-N, ... 0, psi_{i}, omega_{i}, phi_{i+1} or 0, tau_{i},...
    u_2 = U[:,2:,:] # N-C, CA-N, C-CA, ...

    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_2), dim=-1)
    
    cosD = (n_0 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    
    v = _normalize(torch.cross(n_0, n_1), dim=-1)
    D = torch.sign((-v* u_1).sum(-1)) * torch.acos(cosD) # TODO: sign
    
    return D


def _dihedrals(X, dihedral_type=0, eps=1e-7):
    B, N, _, _ = X.shape
    # psi, omega, phi
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) # ['N', 'CA', 'C', 'O']
    D = cal_dihedral(X)
    D = F.pad(D, (1,2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3)) 
    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    # alpha, beta, gamma
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:,:-2,:] # CA-N, C-CA, N-C,...
    u_1 = U[:,1:-1,:] # C-CA, N-C, CA-N, ...
    cosD = (u_0*u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.acos(cosD)
    D = F.pad(D, (1,2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3))
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    D_features = torch.cat((Dihedral_Angle_features, Angle_features), 2)
    return D_features

def _hbonds(X, E_idx, mask_neighbors, eps=1E-3):
    X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

    X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
    X_atoms['H'] = X_atoms['N'] + _normalize(
            _normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
        +  _normalize(X_atoms['N'] - X_atoms['CA'], -1)
    , -1)

    def _distance(X_a, X_b):
        return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

    def _inv_distance(X_a, X_b):
        return 1. / (_distance(X_a, X_b) + eps)

    U = (0.084 * 332) * (
            _inv_distance(X_atoms['O'], X_atoms['N'])
        + _inv_distance(X_atoms['C'], X_atoms['H'])
        - _inv_distance(X_atoms['O'], X_atoms['H'])
        - _inv_distance(X_atoms['C'], X_atoms['N'])
    )

    HB = (U < -0.5).type(torch.float32)
    neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
    return neighbor_HB

def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def _get_rbf(A, B, E_idx=None, num_rbf=16):
    if E_idx is not None:
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = _rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,:,None,:])**2,-1) + 1e-6) #[B, L, L]
        RBF_A_B = _rbf(D_A_B, num_rbf)
    return RBF_A_B

def _orientations_coarse_gl(X: torch.Tensor, E_idx: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """ Compute coarse-grained orientations of the backbone.
    Inputs:
    * X: (B, L, C, 3) protein coordinates?
    * E_idx: (B, L, K) neighbor indices
    Outputs:
    * O: (B, L, 7) coarse-grained orientations
    """
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)
    
    n_0 = n_0[:,::3,:]
    b_1 = b_1[:,::3,:]
    X = X[:,::3,:]

    O = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    O = O.view(*O.shape[:2], 9)
    # TODO: hypnopump@ why pad?
    O = F.pad(O, (0,0,0,1), 'constant', 0) # [16, 464, 9]

    O_neighbors = gather_nodes(O, E_idx) # [b, n, rd]
    X_neighbors = gather_nodes(X, E_idx) # [b, n, k, d]

    # FIXME: hypnopump@ remove pattern once gather_nodes accepts dim
    O = O.view(*O.shape[:2], 1, 3, 3) # [b, n, 1, r, d]
    O_neighbors = O_neighbors.view(*O_neighbors.shape[:3],3,3) # [b, n, k, r, d]

    dX = X_neighbors - X.unsqueeze(-2) # [16, 464, 30, 3]
    dU = torch.matmul(O, dX.unsqueeze(-1)).squeeze(-1) # [16, 464, 30, 3] 邻居的相对坐标
    R = torch.matmul(O.transpose(-1,-2), O_neighbors)
    feat = torch.cat((_normalize(dU, dim=-1), _quaternions(R)), dim=-1) # 相对方向向量+旋转四元数
    return feat


def _orientations_coarse_gl_tuple(X, E_idx, eps=1e-6):
    """
    Inputs:
    * X: [B, N, 3, 3]
    * E_idx: [B, N, K]
    Outputs:
    * V_direct: [B, N, 3, 3] NCO relpos in CA frame
    * E_direct: [B, N, K, ???]
    * q: [???, 4] quaternion
    """
    V = X.clone()
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)
    # TODO: hypnopump@ compute frames only once per protein
    # FIXME: hypnopump@ unpack and compute frames properly, this is ilegible lol
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)
    
    n_0 = n_0[:,::3,:]
    b_1 = b_1[:,::3,:]
    X = X[:,::3,:]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0,0,0,1), 'constant', 0) # (b, n, 3*3)

    # FIXME: hypnopump@ in batched model
    Q_neighbors = gather_nodes(Q, E_idx) # (b, n, k, 3*3)
    XNCO_neighbors = map(partial(gather_nodes, neighbor_idx=E_idx), X.unbind(dim=-2))

    Q = Q.view(*Q.shape[:2], 1, 3, 3) # (b, n, 1, 3, 3)
    Q_neighbors = Q_neighbors.view(*Q_neighbors.shape[:3], 3, 3) # (b, n, k, 3, 3)

    # FIXME: hypnopump@ embedding relative orientations, but clarify better
    dX = torch.stack(list(XNCO_neighbors), dim=-2) - X[:,:,None,None,:] # (b, n, k, c, d)
    dU = torch.matmul(Q[:,:,:,None,:,:], dX[...,None]).squeeze(-1) # (b, n, k, c, d?) relative coords of neighbors
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K,-1)
    R = torch.matmul(Q.transpose(-1,-2), Q_neighbors)
    q = _quaternions(R)
    # edge_feat = torch.cat((dU, q), dim=-1) # 相对方向向量+旋转四元数
    
    # NCO relpos in frame
    dX_inner = V[:,:,[0,2,3],:] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B,N,-1)
    return V_direct, E_direct, q

def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """ Inputs:
    * edges: ???
    * neighbor_idx: ???
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

# TODO: hypnopump@ accept dimension for gathering
# TODO: hypnopump@ pack dims before and after, index, unpack
def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """ Inputs:
    * nodes: [B, N, D]
    * neighbor_idx: [B, N, K] or [B, N, ]
    """
    b, n, d = nodes.shape
    _, _, k = neighbor_idx.shape

    neighbors_flat = neighbor_idx.view(b, -1)  # (b, (n k))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, d)  # (b, (n k), d)
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)  # (b, (n k), d)
    return neighbor_features.view(b, n, k, -1) # (b, n, k, d)


def _quaternions(R: torch.Tensor) -> torch.Tensor:
    """ ((...), 3, 3) -> ((...), 4) """
    ii, ij, ik, ji, jj, jk, ki, kj, kk = torch.unbind(R.reshape(*R.shape[:2], -1), -1)
    proto_q = torch.stack(
        [
            + ii - jj - kk,
            - ii + jj - kk,
            - ii - jj + kk
        ], dim=-1
    )
    magnitudes = 0.5 * proto_q.add_(1.).abs().sqrt()

    signs = torch.stack([kj - jk, ik - ki, ji - ij], -1).sign()
    xyz = signs * magnitudes
    w = F.relu(1 + diag.sum(-1, keepdim=True)).sqrt() / 2.
    Q = torch.cat((xyz, w), -1)
    return _normalize(Q, dim=-1)
