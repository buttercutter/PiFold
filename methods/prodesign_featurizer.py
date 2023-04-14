import time

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils import _dihedrals, _get_rbf, _orientations_coarse_gl_tuple, gather_nodes


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
    bool_sqmask = sqmask.bool()
    D = th.cdist(X, X, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    D.masked_fill_(~bool_sqmask, 0.0)
    D.add_(sqmask.mul(-1000.0))
    # non-priority for non-neighbors, row-wise
    D.add_((1.0 - sqmask) * D.amax(dim=-1, keepdim=True).detach_().add_(1.0))
    # select top_k neighbours for each atom
    D_neighbors, E_idx = th.topk(D, min(top_k, D.shape[-1]), dim=-1, largest=False)
    return D_neighbors, E_idx


def _get_features(
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
    * X: (B, N, C, D) ???
    * mask: (B, N, ...) ???
    * top_k: int. number of neighbors to link.
    * virtual_num: int. Number of virtual atoms.
    * virtual_atoms: (virtual_num, 3) virtual atom positions.
    * num_rbf: int. Number of RBFs for distance encoding.
    Outputs:
    """
    device = X.device
    mask_bool = mask.bool()
    B, N, _, _ = X.shape
    X_ca = X[..., 1, :]
    D_neighbors, E_idx = _full_dist(X=X_ca, mask=mask, top_k=top_k)

    mask_attend = gather_nodes(mask_bool.unsqueeze(-1), E_idx).squeeze(-1)
    edge_mask_select = lambda x: th.masked_select(x, mask_attend[..., None]).reshape(
        -1, x.shape[-1]
    )
    node_mask_select = lambda x: th.masked_select(x, mask_bool[..., None]).reshape(
        -1, x.shape[-1]
    )

    randn = th.rand(mask.shape, device=X.device).add_(5).abs()
    decoding_order = th.argsort(
        -mask * randn
    )  # 我们的mask=1代表数据可用, 而protein MPP的mask=1代表数据不可用，正好相反
    permutation_matrix_reverse = th.nn.functional.one_hot(
        decoding_order, num_classes=N
    ).float()
    # 计算q已知的情况下, q->p的mask,
    order_mask_backward = th.einsum(
        "ij, biq, bjp->bqp",
        th.tril(th.ones(N, N, device=device)),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend2 = th.gather(order_mask_backward, 2, E_idx)
    mask_1D = mask.view(mask.size(0), mask.size(1), 1)
    mask_bw = (mask_1D * mask_attend2).unsqueeze(-1)
    mask_fw = (mask_1D * (1 - mask_attend2)).unsqueeze(-1)
    mask_bw = edge_mask_select(mask_bw).squeeze()
    mask_fw = edge_mask_select(mask_fw).squeeze()

    # sequence
    S = th.masked_select(S, mask_bool)
    if score is not None:
        score = th.masked_select(score, mask_bool)

    # angle & direction
    V_angles = _dihedrals(X, 0)
    V_angles = node_mask_select(V_angles)

    V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
    V_direct = node_mask_select(V_direct)
    E_direct = edge_mask_select(E_direct)
    E_angles = edge_mask_select(E_angles)

    # distance
    atom_N, atom_Ca, atom_C, atom_O = X[..., :4, :].unbind(-2)

    # 'Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C'


    node_list = ["Ca-N", "Ca-C", "Ca-O", "N-C", "N-O", "O-C"]
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split("-")
        node_dist.append(
            node_mask_select(
                _get_rbf(
                    vars()["atom_" + atom1], vars()["atom_" + atom2], None, num_rbf
                ).squeeze()
            )
        )

    if virtual_num > 0:
        virtual_atoms = F.normalize(virtual_atoms, dim=-1)
        # FIXME: hypnopump@ do in batched mode!
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = th.cross(b, c, dim=-1)
        # frame = th.stack([a, b, c], dim=-1)
        # atom_vs = virtual_atoms @ frame + atom_Ca
        for i in range(virtual_atoms.shape[0]):
            vars()["atom_v" + str(i)] = (
                virtual_atoms[i][0] * a
                + virtual_atoms[i][1] * b
                + virtual_atoms[i][2] * c
                + 1 * atom_Ca
            )

        # FIXME: hypnopump@ do (B, N, V, 3) -> (B, N, V, V, RBF) -> (B, N, (V, V, RBF))
        # FIXME: hypnopump@ will required batch implementations downstream
        rbf = _get_rbf(
            vars()["atom_v" + str(1)],
            vars()["atom_v" + str(0)],
            None,
            num_rbf,
        ).squeeze()
        sele_rbf = node_mask_select(rbf)

        for i in range(virtual_atoms.shape[0]):
            # # true atoms
            for j in range(0, i):
                node_dist.append(
                    node_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(i)],
                            vars()["atom_v" + str(j)],
                            None,
                            num_rbf,
                        ).squeeze()
                    )
                )
                node_dist.append(
                    node_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(j)],
                            vars()["atom_v" + str(i)],
                            None,
                            num_rbf,
                        ).squeeze()
                    )
                )
    V_dist = th.cat(node_dist, dim=-1).squeeze()

    pair_lst = [
        "Ca-Ca",
        "Ca-C",
        "C-Ca",
        "Ca-N",
        "N-Ca",
        "Ca-O",
        "O-Ca",
        "C-C",
        "C-N",
        "N-C",
        "C-O",
        "O-C",
        "N-N",
        "N-O",
        "O-N",
        "O-O",
    ]

    edge_dist = []  # Ca-Ca
    for pair in pair_lst:
        atom1, atom2 = pair.split("-")
        rbf = _get_rbf(vars()["atom_" + atom1], vars()["atom_" + atom2], E_idx, num_rbf)
        edge_dist.append(edge_mask_select(rbf))

    if virtual_num > 0:
        for i in range(virtual_atoms.shape[0]):
            edge_dist.append(
                edge_mask_select(
                    _get_rbf(
                        vars()["atom_v" + str(i)],
                        vars()["atom_v" + str(i)],
                        E_idx,
                        num_rbf,
                    )
                )
            )
            for j in range(0, i):
                edge_dist.append(
                    edge_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(i)],
                            vars()["atom_v" + str(j)],
                            E_idx,
                            num_rbf,
                        )
                    )
                )
                edge_dist.append(
                    edge_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(j)],
                            vars()["atom_v" + str(i)],
                            E_idx,
                            num_rbf,
                        )
                    )
                )

    E_dist = th.cat(edge_dist, dim=-1)

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

    _V = th.cat(h_V, dim=-1)
    _E = th.cat(h_E, dim=-1)

    # edge index
    shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
    src = shift.view(B, 1, 1) + E_idx
    src = th.masked_select(src, mask_attend).view(1, -1)
    dst = shift.view(B, 1, 1) + th.arange(0, N, device=src.device).view(
        1, -1, 1
    ).expand_as(mask_attend)
    dst = th.masked_select(dst, mask_attend).view(1, -1)
    E_idx = th.cat((dst, src), dim=0).long()

    decoding_order = (
        node_mask_select((decoding_order + shift.view(-1, 1)).unsqueeze(-1))
        .squeeze()
        .long()
    )

    # 3D point, (B, N, C, 3) -> (masked(B N), C, 3)
    batch_id, chainlen_id = mask.nonzero().transpose(-1, -2)  # index of non-zero values
    X = X[batch_id, chainlen_id]

    return X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order
