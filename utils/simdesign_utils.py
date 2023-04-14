from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from functools import partial
from typing import Optional

"""
Notation:
    - B: Batch size
    - N: Number of AAs
    - L: number or nodes in the graph
    - E: Number of edges
    - K: K-nearest neighbors
    - C: Chain atoms per residue
    - R: Rotation basis vectors
    - D: Dimensions
"""


safe_cdist = partial(th.cdist, compute_mode="donot_use_mm_for_euclid_dist")


# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
def nan_to_num(tensor: torch.Tensor, nan=0.0) -> torch.Tensor:
    return tensor.masked_fill_(torch.isnan(tensor), nan)


def _normalize(tensor: torch.Tensor, dim=-1) -> torch.Tensor:
    return nan_to_num(F.normalize(tensor, dim=dim))


def _hbonds(
    X: torch.Tensor,
    E_idx: torch.Tensor,
    mask_neighbors: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Inputs:
    * X: (B, N, C, D)
    * E_idx: (B, N, K)
    * mask_neighbors: (B, N, N) ???
    Outputs: ???
    """
    N_atoms, CA_atoms, C_atoms, O_atoms = torch.unbind(X, dim=-2)  # (b, n, d)
    C_prev = F.pad(C_atoms[:, 1:, :], (0, 0, 0, 1), "constant", 0)
    H_atoms = N_atoms + _normalize(
        _normalize(N_atoms - C_prev, -1) + _normalize(N_atoms - CA_atoms, -1), -1
    )
    U = (0.084 * 332) * (
        safe_cdist(O_atoms, N_atoms).add_(eps).reciprocal_()
        + safe_cdist(C_atoms, N_atoms).add_(eps).reciprocal_()
        - safe_cdist(O_atoms, H_atoms).add_(eps).reciprocal_()
        - safe_cdist(C_atoms, H_atoms).add_(eps).reciprocal_()
    )
    HB = (U < -0.5).type(torch.float32)  # (b, n, n)
    neighbor_HB = mask_neighbors * batched_index_select(HB, E_idx, dim=-1)  # (b, n, k)
    return neighbor_HB


def _rbf(
    D: torch.Tensor, num_rbf: int, D_min: float = 0.0, D_max: float = 20.0
) -> torch.Tensor:
    """(...,) -> (..., num_rbf)"""
    D_mu = torch.linspace(D_min, D_max, num_rbf, device=D.device, dtype=D.dtype)
    D_mu = D_mu.view(*(1,) * D.ndim, -1)
    D_sigma = (D_max - D_min) / num_rbf
    return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))


def _get_rbf(
    A: torch.Tensor,
    B: torch.Tensor,
    E_idx: Optional[torch.Tensor] = None,
    num_rbf: int = 16,
    eps: float = 1e-6
) -> torch.Tensor:
    """(B, L1, D), (D, L2, D) -> (B, L1, L2, num_rbf)
    Optionally supports edges in the form of (B, L1, K) -> (B, L1, K, num_rbf)
    """

    if E_idx is not None:
        D_A_B = safe_cdist(A, B)  # [B, L1, L2]
        D_A_B = batched_index_select(D_A_B, E_idx, dim=-1)  # [B, L1, K]
    else:
        D_A_B = (A - B).square().sum(-1).add_(eps).sqrt()[..., None]  # [B, L1, 1]

    RBF_A_B = _rbf(D_A_B, num_rbf)  # [B, L, X] (X = 1 or K)
    return RBF_A_B


# TODO: fuse with orientations to reuse most computations
def _dihedrals(
    X: torch.Tensor, dihedral_type: int = 0, eps: float = 1e-7
) -> torch.Tensor:
    """X is (B, (N C), D)"""
    B, N, _, _ = X.shape

    # Relative positions
    # psi, omega, phi
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)  # ['N', 'CA', 'C', 'O']
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]  # CA-N, C-CA, N-C,...
    u_1 = U[
        :, 1:-1, :
    ]  # C-CA, N-C, CA-N, ... 0, psi_{i}, omega_{i}, phi_{i+1} or 0, tau_{i},...
    u_2 = U[:, 2:, :]  # N-C, CA-N, C-CA, ...

    # Dihedrals: (b, n, c, 2)
    u1u2_cross = th.cross(u_1, u_2, dim=-1)

    D = th.atan2(
        (th.norm(u_0, dim=-1, keepdim=True) * u_1 * u1u2_cross).sum(-1),
        (th.cross(u_0, u_1, dim=-1) * u1u2_cross).sum(-1),
    )
    D.masked_fill_(D.isnan(), 0.0)
    D = F.pad(D, (1, 2), "constant", 0)  # (B, (N C)-1)
    Dih = D.view(D.shape[0], D.shape[1] // 3, 3)  # (B, N, C)

    # alpha, beta, gamma
    cosD = (
        (u_0 * u_1).sum(-1).clamp_(min=-1 + eps, max=1 - eps)
    )  # alpha_{i}, gamma_{i}, beta_{i+1}
    D = torch.acos(cosD)
    D = F.pad(D, (1, 2), "constant", 0)
    Ang = D.view(D.shape[0], D.shape[1] // 3, 3)

    return torch.cat((Dih.cos(), Dih.sin(), Ang.cos(), Ang.sin()), -1)  # (b, n, c*2*2)


def _orientations_coarse_gl_tuple(
    X: torch.Tensor, E_idx: torch.Tensor, eps=1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Relative vectors of neighbors under current reference frame.
    1. compute frames (Q)
    2. Compute Reldists (dX)
    3. Compute Relorients (dU)
    4. Normalize (E_direct)
    5. ???

    Inputs:
    * X: [B, N, C, 3]
    * E_idx: [B, N, K]
    Outputs:
    * V_direct: [B, N, 3, 3] NCO relpos in CA frame ???
    * E_direct: [B, N, K, ???]
    * q: [???, 4] quaternion
    """
    V = X.clone()
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)  # (b, (n c), 3)
    # TODO: hypnopump@ compute frames only once per protein
    # FIXME: hypnopump@ unpack and compute frames properly, this is ilegible lol
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:, :-2, :], U[:, 1:-1, :]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)

    n_0 = n_0[:, ::3, :]
    b_1 = b_1[:, ::3, :]
    X = X[:, ::3, :]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0, 0, 0, 1), "constant", 0)  # (b, n, 3*3)

    # FIXME: hypnopump@ in batched model
    Q_neighbors = gather_nodes(Q, E_idx)  # (b, n(?), k, 3*3)
    XNCO_neighbors = map(partial(gather_nodes, neighbor_idx=E_idx), V.unbind(dim=-2))

    Q = Q.view(*Q.shape[:2], 1, 3, 3)  # (b, n, 1, 3, 3)
    Q_neighbors = Q_neighbors.view(*Q_neighbors.shape[:3], 3, 3)  # (b, n, k, 3, 3)

    # FIXME: hypnopump@ embedding relative orientations, but clarify better
    dX = (
        torch.stack(list(XNCO_neighbors), dim=-2) - X[:, :, None, None, :]
    )  # (b, n, k, c, d)
    dU = torch.matmul(Q[:, :, :, None, :, :], dX[..., None]).squeeze(
        -1
    )  # (b, n, k, c, d?) relative coords of neighbors
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K, -1)
    R = torch.matmul(Q.transpose(-1, -2), Q_neighbors)
    q = _quaternions(R)
    # edge_feat = torch.cat((dU, q), dim=-1) # 相对方向向量+旋转四元数

    # NCO relpos in frame
    dX_inner = V[:, :, [0, 2, 3], :] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B, N, -1)
    return V_direct, E_direct, q


def batched_index_select(values: th.Tensor, indices: th.Tensor, dim=1) -> th.Tensor:
    """Batched select from values. Assumes the first `dim` dims are the same,
    until the indexing dimension. Then indices and values might difer.
    Inputs:
    * values: (..., n, vvv) th.Tensor. vvv is any number of dims
    * indices: (..., k, iii) th.Tensor. iii is any number of dims
    Outputs: (..., k, iii, vvv)
    """
    vdim = dim if dim > 0 else values.ndim + dim
    values_shape, indices_shape = map(lambda x: list(x.shape), (values, indices))
    extra_value_dims, extra_indices_dims = map(
        lambda x: x[vdim + 1 :], (values_shape, indices_shape)
    )
    # (..., k, iii) -> (..., k, iii, vvv)
    indices = indices[(..., *((None,) * len(extra_value_dims)))]
    # (..., n, vvv) -> (..., n, iii, vvv)
    values = values[
        (*((slice(None),) * (dim + 1)), *((None,) * len(extra_indices_dims)), ...)
    ]
    # expand to match shapes except dim
    indices = indices.expand(*((-1,) * len(indices_shape)), *extra_value_dims)
    values = values.expand(
        *(-1,) * (vdim + 1), *extra_indices_dims, *(-1,) * len(extra_value_dims)
    )
    return values.gather(dim, indices)


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Inputs:
    * edges: ???
    * neighbor_idx: ???
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


# TODO: hypnopump@ accept dimension for gathering
# TODO: hypnopump@ pack dims before and after, index, unpack
def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Inputs:
    * nodes: [B, N, D]
    * neighbor_idx: [B, N, K] or [B, N, ]
    """
    b, n, d = nodes.shape
    _, _, k = neighbor_idx.shape

    neighbors_flat = neighbor_idx.view(b, -1)  # (b, (n k))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, d)  # (b, (n k), d)
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)  # (b, (n k), d)
    return neighbor_features.view(b, n, k, -1)  # (b, n, k, d)


def _quaternions(R: th.Tensor) -> th.Tensor:
    """((...), 3, 3) -> ((...), 4)"""
    ii, ij, ik, ji, jj, jk, ki, kj, kk = th.unbind(R.reshape(*R.shape[:-2], -1), -1)
    qi = +ii - jj - kk
    qj = -ii + jj - kk
    qk = -ii - jj + kk
    proto_q = th.stack((qi, qj, qk), dim=-1)
    magnitudes = proto_q.add_(1.0).abs().sqrt().mul_(0.5)

    signs = th.stack([kj - jk, ik - ki, ji - ij], -1).sign()
    xyz = signs * magnitudes
    w = F.relu(1 + ii + jj + kk).sqrt().mul_(0.5)
    Q = th.cat((xyz, w[..., None]), -1)
    return _normalize(Q, dim=-1)
