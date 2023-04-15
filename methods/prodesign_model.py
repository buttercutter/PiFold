import logging
import time
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm.layer_norm import LayerNorm as sparseLayerNorm

from utils import _dihedrals, _get_rbf, _orientations_coarse_gl_tuple, gather_nodes

from .common import Linear
from .prodesign_featurizer import _full_dist, _get_features_dense, _get_features_sparse
from .prodesign_module import *

logger = logging.getLogger(__name__)


class NormWrapper(nn.Module):
    def __init__(self, *args, norm_choice: str, **kwargs):
        """Wraps different norm choices and considerations.
        Inputs:
        * norm_choice: str. One of ["batchnorm", "layernorm", "none"]
        """
        super().__init__()
        self.norm_choice = norm_choice

        self.norm = nn.Identity()
        if norm_choice == "batchnorm":
            self.norm = nn.BatchNorm1d(*args, **kwargs)
        elif norm_choice == "layernorm":
            self.norm = sparseLayerNorm(*args, **kwargs)
        else:
            logger.warning(
                f"Coyld not recognize norm choice {norm_choice}. Setting Identity"
            )

    def forward(
        self,
        x: th.Tensor,
        mask: Optional[th.Tensor] = None,
        batch_index: Optional[th.Tensor] = None,
        mode: str = "sparse",
    ) -> th.Tensor:
        """Runs norm of choice dealing with custom needs for aggregation / masking.
        Inputs:
        * x: th.Tensor. Input features. As in any norm layer.
        * mask: th.Tensor. Mask of valid nodes. Needed for batchnorm in dense mode.
        """
        if self.norm_choice == "batchnorm":
            if mode == "dense":
                assert (
                    mask is not None
                ), f"Dense implementation of {self.norm_choice} requires a mask."
                x_ = x.clone()
                x_[mask] = self.norm(x_[mask])
                return x_

        elif self.norm_choice == "layernorm":
            if mode == "sparse":
                assert (
                    batch_index is not None
                ), f"Sparse implementation of {self.norm_choice} requires a batch index."
                return self.norm(x, batch_index)
            else:
                return F.layer_norm(
                    x, [*x.shape[-1:]], self.norm.weight, self.norm.bias, self.norm.eps
                )

        return self.norm(x)


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
        self.norm_class = partial(NormWrapper, norm_choice=args.norm_choice)

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
        self.norm_nodes = self.norm_class(node_features)
        self.norm_edges = self.norm_class(edge_features)

        # TODO: hypnopump@ consider LayerNorm (if dense) or OnlineNorm ???
        self.W_v_act = nn.GELU()
        self.W_v_norm1 = self.norm_class(hidden_dim)
        self.W_v_norm2 = self.norm_class(hidden_dim)
        self.W_v = nn.ModuleList(
            [
                Linear(node_features, hidden_dim, bias=True, init="relu"),
                Linear(node_features, hidden_dim, bias=True, init="relu"),
                Linear(node_features, hidden_dim, bias=True),
            ]
        )

        self.W_e = Linear(edge_features, hidden_dim, bias=True)
        # FIXME: hypnopump@ not used ???
        self.W_f = Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(
            hidden_dim,
            num_encoder_layers,
            dropout,
            num_heads=args.num_heads,
            checkpoint=args.checkpoint,
            norm_class=self.norm_class,
        )

        self.decoder = MLPDecoder(hidden_dim, norm_class=self.norm_class)
        self._init_params()

        self.encode_t = 0
        self.decode_t = 0

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
        mode: str = "sparse",
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ):
        t1 = time.time()
        # node and edge embeddings
        h_V = self.norm_nodes(
            self.node_embedding(h_V), batch_index=batch_id, mask=node_mask, mode=mode
        )
        h_V = self.W_v_norm1(
            self.W_v_act(self.W_v[0](h_V)),
            batch_index=batch_id,
            mask=node_mask,
            mode=mode,
        )
        h_V = self.W_v_norm2(
            self.W_v_act(self.W_v[1](h_V)),
            batch_index=batch_id,
            mask=node_mask,
            mode=mode,
        )
        h_V = self.W_v[2](h_V)
        e_b_idx = batch_id[P_idx[0]] if mode == "sparse" else batch_id
        h_P = self.W_e(
            self.norm_edges(
                self.edge_embedding(h_P), batch_index=e_b_idx, mask=edge_mask, mode=mode
            )
        )

        h_V, h_P = self.encoder(
            h_V,
            h_P,
            P_idx,
            batch_id,
            mode=mode,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )
        t2 = time.time()

        log_probs, logits = self.decoder(
            h_V, batch_id, mode=mode, node_mask=node_mask, edge_mask=edge_mask
        )

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

    def _get_features(
        self,
        S: torch.Tensor,
        score: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        mode: str = "sparse",
    ) -> list[torch.Tensor]:
        """Gets features as it needs the virtual atoms."""
        _get_features_func = (
            _get_features_sparse if mode == "sparse" else _get_features_dense
        )
        return _get_features_func(
            S=S,
            score=score,
            X=X,
            mask=mask,
            # FIXME: hypnopump@ `virtual_atoms` prevents us from moving to pure CPU setup
            virtual_atoms=self.virtual_atoms,
            top_k=self.top_k,
            num_rbf=self.num_rbf,
            virtual_num=self.args.virtual_num,
            node_dist=self.args.node_dist,
            node_angle=self.args.node_angle,
            node_direct=self.args.node_direct,
            edge_dist=self.args.edge_dist,
            edge_angle=self.args.edge_angle,
            edge_direct=self.args.edge_direct,
        )
