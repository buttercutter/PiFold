import time

import torch
import torch.nn as nn
from torch_geometric.nn.norm.layer_norm import LayerNorm as sparseLayerNorm

from utils import _dihedrals, _get_rbf, _orientations_coarse_gl_tuple, gather_nodes

from .common import Linear
from .prodesign_module import *


class ProDesign_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """Graph labeling network"""
        super(ProDesign_Model, self).__init__()
        self.args = args
        node_features = args.node_features
        edge_features = args.edge_features
        hidden_dim = args.hidden_dim
        dropout = args.dropout
        num_encoder_layers = args.num_encoder_layers
        self.top_k = args.k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16
        self.norm_class = (
            nn.BatchNorm1d
            if args.norm == "batchnorm"
            else (sparseLayerNorm if args.norm == "layernorm" else nn.Identity)
        )

        # prior_matrix = [
        #     [-0.58273431, 0.56802827, -0.54067466],
        #     [0.0       ,  0.83867057, -0.54463904],
        #     [0.01984028, -0.78380804, -0.54183614],
        # ]

        # self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix)[:self.args.virtual_num,:])

        # TODO: hypnopump@ init to smaller std ???
        self.virtual_atoms = nn.Parameter(torch.rand(self.args.virtual_num, 3))

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num > 0:
                pair_num += self.args.virtual_num * (self.args.virtual_num - 1)
            node_in += pair_num * self.num_rbf
        if self.args.node_angle:
            node_in += 12
        if self.args.node_direct:
            node_in += 9

        edge_in = 0
        if self.args.edge_dist:
            pair_num = 16

            if self.args.virtual_num > 0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num * (self.args.virtual_num - 1)
            edge_in += pair_num * self.num_rbf
        if self.args.edge_angle:
            edge_in += 4
        if self.args.edge_direct:
            edge_in += 12

        self.node_embedding = Linear(node_in, node_features, bias=True)
        self.edge_embedding = Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = self.norm_class(node_features, affine=False)
        self.norm_edges = self.norm_class(edge_features, affine=False)

        # TODO: hypnopump@ consider LayerNorm (if dense) or OnlineNorm ???
        self.W_v = nn.Sequential(
            Linear(node_features, hidden_dim, bias=True, init="relu"),
            nn.GELU(),
            self.norm_class(hidden_dim, affine=False),
            Linear(hidden_dim, hidden_dim, bias=True, init="relu"),
            nn.GELU(),
            self.norm_class(hidden_dim, affine=False),
            Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.W_e = Linear(edge_features, hidden_dim, bias=True)
        # FIXME: hypnopump@ not used ???
        self.W_f = Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(
            hidden_dim,
            num_encoder_layers,
            dropout,
            checkpoint=args.checkpoint,
            norm_class=self.norm_class,
        )

        self.decoder = MLPDecoder(hidden_dim, norm_class=self.norm_class)
        self._init_params()

        self.encode_t = 0
        self.decode_t = 0

        if self.args.max_num_iters > 1:
            self.recycle_V = self.norm_class(hidden_dim)
            self.recycle_E = self.norm_class(hidden_dim)
            self.recycle_P = nn.Sequential(
                nn.Softmax(),
                self.norm_class(self.decoder.vocab, affine=False),
                Linear(self.decoder.vocab, hidden_dim, bias=True),
            )

    def recycling(self, h_V, h_P, logits):
        """Recycles last dim's node (+ probs), edge and probs embedding."""
        return self.recycle_V(h_V) + self.recycle_P(logits), self.recycle_E(h_P)

    def forward(
        self,
        h_V,
        h_P,
        P_idx,
        batch_id,
        S=None,
        AT_test=False,
        mask_bw=None,
        mask_fw=None,
        decoding_order=None,
        return_logit=False,
        num_iters: int = 1,
    ):
        t1 = time.time()
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))

        is_grad_enabled = torch.is_grad_enabled()
        for i in range(num_iters):
            # only enable gradient for the last cycle
            with th.set_grad_enabled(i == num_iters - 1 and is_grad_enabled):
                if i != 0:
                    h_V_prev, h_P_prev = self.recycling(h_V, h_P, logits)
                    h_V = h_V_prev + h_V
                    h_P = h_P_prev + h_P

                h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)
                t2 = time.time()

                log_probs, logits = self.decoder(h_V, batch_id)

                t3 = time.time()

            self.encode_t += t2 - t1
            self.decode_t += t3 - t2

        if return_logit == True:
            return log_probs, logits
        return log_probs

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == "virtual_atoms":
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = (1.0 - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(
            D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _get_features(self, S, score, X, mask):
        device = X.device
        mask_bool = mask == 1
        B, N, _, _ = X.shape
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(
            x, mask_attend.unsqueeze(-1)
        ).reshape(-1, x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(
            x, mask_bool.unsqueeze(-1)
        ).reshape(-1, x.shape[-1])

        randn = torch.rand(mask.shape, device=X.device) + 5
        decoding_order = torch.argsort(
            -mask * (torch.abs(randn))
        )  # 我们的mask=1代表数据可用, 而protein MPP的mask=1代表数据不可用，正好相反
        mask_size = mask.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        # 计算q已知的情况下, q->p的mask,
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend2 = torch.gather(order_mask_backward, 2, E_idx)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1])
        mask_bw = (mask_1D * mask_attend2).unsqueeze(-1)
        mask_fw = (mask_1D * (1 - mask_attend2)).unsqueeze(-1)
        mask_bw = edge_mask_select(mask_bw).squeeze()
        mask_fw = edge_mask_select(mask_fw).squeeze()

        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X, 0)
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N, atom_Ca, atom_C, atom_O = X.unbind(dim=-2)[:4]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.args.virtual_num > 0:
            virtual_atoms = self.virtual_atoms / torch.norm(
                self.virtual_atoms, dim=1, keepdim=True
            )
            for i in range(self.virtual_atoms.shape[0]):
                vars()["atom_v" + str(i)] = (
                    virtual_atoms[i][0] * a
                    + virtual_atoms[i][1] * b
                    + virtual_atoms[i][2] * c
                    + 1 * atom_Ca
                )

        node_list = ["Ca-N", "Ca-C", "Ca-O", "N-C", "N-O", "O-C"]
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split("-")
            node_dist.append(
                node_mask_select(
                    _get_rbf(
                        vars()["atom_" + atom1],
                        vars()["atom_" + atom2],
                        None,
                        self.num_rbf,
                    ).squeeze()
                )
            )

        if self.args.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                # # true atoms
                for j in range(0, i):
                    node_dist.append(
                        node_mask_select(
                            _get_rbf(
                                vars()["atom_v" + str(i)],
                                vars()["atom_v" + str(j)],
                                None,
                                self.num_rbf,
                            ).squeeze()
                        )
                    )
                    node_dist.append(
                        node_mask_select(
                            _get_rbf(
                                vars()["atom_v" + str(j)],
                                vars()["atom_v" + str(i)],
                                None,
                                self.num_rbf,
                            ).squeeze()
                        )
                    )
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

        # TODO: hypnopump@ consider replace with itertools.product + '-'.join()
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

        # FIXME: hypnopump@ this is disgusting! reformulate in a proper dict!
        edge_dist = []  # Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split("-")
            rbf = _get_rbf(
                vars()["atom_" + atom1], vars()["atom_" + atom2], E_idx, self.num_rbf
            )
            edge_dist.append(edge_mask_select(rbf))

        if self.args.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(
                    edge_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(i)],
                            vars()["atom_v" + str(i)],
                            E_idx,
                            self.num_rbf,
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
                                self.num_rbf,
                            )
                        )
                    )
                    edge_dist.append(
                        edge_mask_select(
                            _get_rbf(
                                vars()["atom_v" + str(j)],
                                vars()["atom_v" + str(i)],
                                E_idx,
                                self.num_rbf,
                            )
                        )
                    )

        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        h_V = []
        if self.args.node_dist:
            h_V.append(V_dist)
        if self.args.node_angle:
            h_V.append(V_angles)
        if self.args.node_direct:
            h_V.append(V_direct)

        h_E = []
        if self.args.edge_dist:
            h_E.append(E_dist)
        if self.args.edge_angle:
            h_E.append(E_angles)
        if self.args.edge_direct:
            h_E.append(E_direct)

        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)

        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B, 1, 1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1, -1)
        dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(
            1, -1, 1
        ).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1, -1)
        E_idx = torch.cat((dst, src), dim=0).long()

        decoding_order = (
            node_mask_select((decoding_order + shift.view(-1, 1)).unsqueeze(-1))
            .squeeze()
            .long()
        )

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:, 0], sparse_idx[:, 1], :, :]
        batch_id = sparse_idx[:, 0]

        return X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order
