import time

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from utils import _dihedrals, _get_rbf, _orientations_coarse_gl_tuple, gather_nodes, batched_index_select


def _full_dist(
    X: th.Tensor, mask: th.Tensor, top_k: int = 30, eps: float = 1e-6
) -> tuple[th.Tensor, th.Tensor]:
    """Returns distances of neighbours (by mask).
    Inputs:
    * X: th.Tensor. shape (B, N, 3). Coordinates of atoms.
    * mask: th.Tensor. shape (B, N). Mask of atoms.
    * top_k: int. Number of neighbours to link.
    * eps: float. Small number to avoid division by zero.
    Outputs:
    * D_neighbors: th.Tensor. shape (B, N, top_k). Distances of neighbours.
    * E_idx: th.Tensor. shape (B, N, top_k). Indices of neighbours.
    """
    # TODO: hypnopump@ rethink this to mask non-neighs as maxdist + 1: easier.
    # (B, N, N)
    sqmask = mask[..., None, :] * mask[..., None]
    sqcdist = th.sum((X[..., None, :] - X[..., None, :, :]).square(), dim=-1)
    D = th.sqrt(sqcdist.add_(eps))
    # bias by sequence distance: further apart = higher distance
    pair_idist = th.arange(D.shape[-1], device=D.device, dtype=D.dtype)
    pair_idist = (pair_idist[None] - pair_idist[:, None]).abs().mul_(1e-5)
    D = D + pair_idist[(*(None,) * (D.ndim - 2), ...)]
    # increase dist of non-neighbors to max for node
    D.add_((1.0 - sqmask) * D.amax(dim=-1, keepdim=True).detach_().add_(1.0))
    D_neighbors, E_idx = th.topk(D, min(top_k, D.shape[-1]), dim=-1, largest=False)
    return D_neighbors, E_idx


def _get_features_dense(
    S: th.Tensor,
    score: th.Tensor,
    X: th.Tensor,
    mask: th.Tensor,
    top_k: int,
    virtual_num: int,
    virtual_atoms: th.Tensor,
    num_rbf: int,
    node_dist: bool,
    node_angle: bool,
    node_direct: bool,
    edge_dist: bool,
    edge_angle: bool,
    edge_direct: bool,
) -> list[th.Tensor]:
    """Get the features for the model.
    Inputs:
    * S: ???
    * score: ???
    * X: (B, N, C=4, D) coordinates of BB atoms
    * mask: (B, N) float mask indicating valid (present, resolved AAs)
    * top_k: int. number of neighbors to link.
    * virtual_num: int. Number of virtual atoms.
    * virtual_atoms: (virtual_num, 3) virtual atom positions.
    * num_rbf: int. Number of RBFs for distance encoding.
    Outputs: (B, N, D) data.
    """
    device = X.device
    B, N, _, _ = X.shape
    X_ca = X[..., 1, :]  # (B, N, D)
    mask_bool = mask.bool()  # (B, N)
    # (B, N, K) dists, (B, N, K) edge idx
    D_neighbors, E_idx = _full_dist(X=X_ca, mask=mask, top_k=top_k)

    # (B, N, K)
    mask_attend = batched_index_select(mask_bool, E_idx, dim=-1)
    mask_attend = (mask[..., None] * mask_attend).bool()

    # (B, N)
    randn = th.rand(mask.shape, device=X.device).add_(5).abs()
    # Our mask=1 represents available data, vs the protein MPP's mask=1 which represents unavailable data
    decoding_order = th.argsort(-mask * randn)
    # Calc mask from q to p, given the known q
    permutation_matrix_reverse = th.nn.functional.one_hot(decoding_order, num_classes=N).float()
    order_mask_backward = th.einsum(
        "ij, biq, bjp->bqp",
        th.tril(th.ones(N, N, device=device, dtype=permutation_matrix_reverse.dtype)),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend2 = th.gather(order_mask_backward, 2, E_idx)
    mask_1D = mask[..., None]
    mask_bw = (mask_1D * mask_attend2)
    mask_fw = (mask_1D * (1 - mask_attend2))

    # angle & direction
    V_angles = _dihedrals(X, 0)
    V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)

    # Locate virtual atoms
    if virtual_num > 0:
        virtual_atoms = F.normalize(virtual_atoms, dim=-1)
        atom_N, atom_Ca, atom_C, atom_O = X[..., :4, :].unbind(-2)
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = th.cross(b, c, dim=-1)
        # (b, n, f=3, d)
        frame = th.stack([a, b, c], dim=-2)
        # (v d), (b, n, f=3, d) -> (b n v d)
        atom_vs = virtual_atoms[(*(None,) * (frame.ndim - virtual_atoms.ndim), ...)]
        atom_vs = atom_vs @ frame + atom_Ca[..., None, :]

    # Node distances
    # Ca-N, Ca-C, Ca-O, N-C, N-O, O-C

    # (b n c d) -> (b n c c d) -> (b n triu(c c) rbf)
    row_aidxs, cols_aidxs = torch.triu_indices(
        X.shape[-2], X.shape[-2], offset=1, device=X.device
    )
    anode_rbf = _get_rbf(X[..., None, :], X[..., None, :, :], None, num_rbf).squeeze(-2)
    V_dist = [*anode_rbf[..., row_aidxs, cols_aidxs, :].unbind(dim=-2)]

    if virtual_num > 0:
        # FIXME: hypnopump@ original code passes tril + triu -> but its symmetric. reuduce to triu
        # row_vidxs, cols_vidxs = torch.triu_indices(atom_vs.shape[-2], atom_vs.shape[-2], offset=1, device=X.device)
        row_vidxs, cols_vidxs = (
            ~torch.eye(atom_vs.shape[-2], dtype=torch.bool)
        ).nonzero(as_tuple=True)

        # (b n v d) -> (b n v v d) -> (b n triu(v v) rbf)
        vnode_rbf = _get_rbf(
            atom_vs[..., None, :], atom_vs[..., None, :, :], None, num_rbf
        ).squeeze(-2)
        V_dist += [*vnode_rbf[..., row_vidxs, cols_vidxs, :].unbind(dim=-2)]

    # p x (b n rbf) -> (b, n, (p rbf))
    V_dist = th.cat(V_dist, dim=-1)

    # Batched Edge distance encoding
    # CA-CA, CA-C, C-CA, CA-N, N-CA, CA-O, O-CA, C-C, C-N, N-C, C-O, O-C, N-N, N-O, O-N, O-O

    # (b n c d), (b, n, k) -> (c, b, n, d), (c, c, b, n, k)
    X_chain = X.transpose(-2, -3).transpose(-3, -4)
    E_idx_chain = E_idx[None, None].expand(
        X_chain.shape[0], X_chain.shape[0], *(-1,) * E_idx.ndim
    )

    # (c b n d) -> (c c b n n) -> (c c b n k) -> c^2 x ((b n), k)
    pair_rbfs = _get_rbf(X_chain[None], X_chain[:, None], E_idx_chain, num_rbf)
    E_dist = [*pair_rbfs.reshape(-1, *pair_rbfs.shape[2:]).unbind(dim=0)]

    if virtual_num > 0:
        # (b n v d), (b, n, k) -> (v, b, n, d), (v v, b, n, k)
        X_virt = atom_vs.transpose(-2, -3).transpose(-3, -4)
        E_idx_virt = E_idx[None, None].expand(
            X_virt.shape[0], X_virt.shape[0], *(-1,) * E_idx.ndim
        )

        # keep diagonal here bc (n n) is not 0
        # (v b n d) -> (v v b n n) -> (v v b n k) -> v^2 x (b n k)
        pair_rbfs = _get_rbf(X_virt[None], X_virt[:, None], E_idx_virt, num_rbf)
        E_dist += [*pair_rbfs.reshape(-1, *pair_rbfs.shape[2:]).unbind(dim=0)]

    # q x (b n k rbf) -> (b, n, k, (q rbf))
    E_dist = th.cat(E_dist, dim=-1)

    # stack node, edge feats
    h_V = []
    if node_dist:
        h_V.append(V_dist)
    if node_angle:
        h_V.append(V_angles)
    if node_direct:
        h_V.append(V_direct)

    h_E = []
    if edge_dist:
        h_E.append(E_dist)
    if edge_angle:
        h_E.append(E_angles)
    if edge_direct:
        h_E.append(E_direct)

    _V = th.cat(h_V, dim=-1) # (B, N, D)
    _E = th.cat(h_E, dim=-1) # (B, N, K, D)
    batch_id = th.arange(X.shape[0], device=X.device)[..., None].expand_as(mask) # (B, N)
    return X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order, mask_bool, mask_attend


def _get_features_sparse(
    S: th.Tensor,
    score: th.Tensor,
    X: th.Tensor,
    mask: th.Tensor,
    top_k: int,
    virtual_num: int,
    virtual_atoms: th.Tensor,
    num_rbf: int,
    node_dist: bool,
    node_angle: bool,
    node_direct: bool,
    edge_dist: bool,
    edge_angle: bool,
    edge_direct: bool,
) -> list[th.Tensor]:
    """Get the features for the model.
    Inputs:
    * S: ???
    * score: ???
    * X: (B, N, C=4, D) coordinates of BB atoms
    * mask: (B, N) float mask indicating valid (present, resolved AAs)
    * top_k: int. number of neighbors to link.
    * virtual_num: int. Number of virtual atoms.
    * virtual_atoms: (virtual_num, 3) virtual atom positions.
    * num_rbf: int. Number of RBFs for distance encoding.
    Outputs: (mask(B N), ...) for node data and (mask(B N K), ...) for edge data
    """
    X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order, node_mask, edge_mask = _get_features_dense(
        S, score, X, mask, top_k, virtual_num, virtual_atoms, num_rbf,
        node_dist, node_angle, node_direct, edge_dist, edge_angle, edge_direct
    )
    mask_bool, mask_attend = node_mask, edge_mask
    B, N = mask_bool.shape

    # Get edge idxs
    # (b, n) -> (b, 1, 1), (b n k) -> ... -> (2, (b n k)) + offsets
    shift = (mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1))[..., None, None].long()
    src = shift + E_idx
    src = th.masked_select(src, mask_attend).view(1, -1)
    dst = shift + th.arange(0, N, device=src.device)[None, :, None].expand_as(
        mask_attend
    )
    dst = th.masked_select(dst, mask_attend).view(1, -1)
    E_idx = th.cat((dst, src), dim=0).long()

    # (b n ...) -> (mask(b n) ...)
    X, S, score, _V, batch_id, decoding_order = map(lambda x: x[mask_bool], (X, S, score, _V, batch_id, decoding_order))
    # (b n k ...) -> (mask(b n k) ...) for edges
    _E2, mask_bw, mask_fw = map(lambda x: x[mask_attend], (_E, mask_bw, mask_fw))

    return X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order, mask_bool, mask_attend