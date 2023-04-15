import time

import torch
import torch.nn as nn

from utils import _dihedrals, _get_rbf, _orientations_coarse_gl_tuple, gather_nodes

from .common import Linear
from .prodesign_module import *
from .prodesign_featurizer import _full_dist, _get_features_sparse, _get_features_dense


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
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        # TODO: hypnopump@ consider LayerNorm (if dense) or OnlineNorm ???
        self.W_v = nn.Sequential(
            Linear(node_features, hidden_dim, bias=True, init="relu"),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            Linear(hidden_dim, hidden_dim, bias=True, init="relu"),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.W_e = Linear(edge_features, hidden_dim, bias=True)
        # FIXME: hypnopump@ not used ???
        self.W_f = Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(
            hidden_dim, num_encoder_layers, dropout, checkpoint=args.checkpoint
        )

        self.decoder = MLPDecoder(hidden_dim)
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
    ):
        if mode != "sparse":
            raise NotImplementedError("Only sparse mode is supported for now")
        t1 = time.time()
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))

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

    def _get_features(self, S: torch.Tensor, score: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, mode: str = "sparse") -> list[torch.Tensor]:
        """ Gets features as it needs the virtual atoms. """
        _get_features_func = _get_features_sparse if mode == "sparse" else _get_features_dense
        return _get_features_func(
            S=S,
            score=score,
            X=X, mask=mask,
            # FIXME: hypnopump@ `virtual_atoms` prevents us from moving to pure CPU setup
            virtual_atoms=self.virtual_atoms,
            top_k=self.top_k,
            num_rbf=self.num_rbf,
            virtual_num = self.args.virtual_num,
            node_dist=self.args.node_dist,
            node_angle=self.args.node_angle,
            node_direct=self.args.node_direct,
            edge_dist=self.args.edge_dist,
            edge_angle=self.args.edge_angle,
            edge_direct=self.args.edge_direct
        )
