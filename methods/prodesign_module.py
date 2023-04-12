from typing import Optional

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

from utils import gather_nodes

from .common import Linear

"""============================================================================================="""
""" Graph Encoder """
"""============================================================================================="""


def get_attend_mask(idx: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Get the mask of the attended nodes.
    Parameters
    ----------
    idx: th.Tensor
        The indices of the attended nodes. Shape: ???
    mask: th.Tensor
        The mask of the nodes. Shape: ???

    Returns
    -------
    mask_attend: th.Tensor shape: ???
    """
    # TODO: write comments in english
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(
        -1
    )  # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
    mask_attend = mask.unsqueeze(-1) * mask_attend  # 自身的mask*邻居节点的mask
    return mask_attend


#################################### node modules ###############################
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=True):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp

        self.W_V = nn.Sequential(
            Linear(num_in, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="glorot"),
        )
        self.Bias = nn.Sequential(
            Linear(num_hidden * 3, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_heads, init="glorot"),
        )
        # FIXME: hypnopump@: only declare if self.output_mlp ??
        self.W_O = Linear(num_hidden, num_hidden, bias=False, init="final")

    def forward(
        self,
        h_V: th.Tensor,
        h_E: th.Tensor,
        center_id: th.Tensor,
        batch_id: th.Tensor,
        dst_idx: Optional[th.Tensor] = None,
    ):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        # (N,d1), (E,d2) -> (E,d1), (E,d2) -> (E,d1+d2) -> (E, heads, 1)
        w = (
            self.Bias(th.cat([h_V[center_id], h_E], dim=-1))
            .view(E, n_heads, 1)
            .mul_(d**0.5)
        )
        # (E, d2) -> (E, heads, d)
        V = self.W_V(h_E).view(-1, n_heads, d)
        # (E, heads, 1)
        attend = scatter_softmax(w.mul_(1 / d**0.5), index=center_id, dim=0)
        # (E, heads, d) * (E, heads, 1) -> (N, heads * d)
        h_V = scatter_sum(attend * V, center_id, dim=0).view([-1, self.num_hidden])

        if self.output_mlp:
            h_V_update = self.W_O(h_V)
        else:
            h_V_update = h_V
        return h_V_update


#################################### edge modules ###############################
class EdgeMLP(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads=None,
        scale=30,
        norm_class: nn.Module = nn.BatchNorm1d,
    ):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = norm_class(num_hidden)
        # TODO: hypnopump@: if not loading published models, turn into MLP ???
        self.W11 = Linear(num_hidden + num_in, num_hidden, bias=True, init="relu")
        self.W12 = Linear(num_hidden, num_hidden, bias=True, init="relu")
        self.W13 = Linear(num_hidden, num_hidden, bias=True, init="final")
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx, dst_idx = edge_idx

        # (N, d1), (E, d2), (N, d1) -> (E, d1+d2+d1)
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        # TODO: hypnopump@: if inference and memory consumption is a problem, use inplace operation
        # TODO: hypnopump@: clarify reisudal better ??? either inside-block or outside-block, not mixed
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


#################################### context modules ###############################
class Context(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads=None,
        scale=30,
        node_context=False,
        edge_context=False,
    ):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context

        self.V_MLP = nn.Sequential(
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden),
        )

        self.V_MLP_g = nn.Sequential(
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="gating"),
            nn.Sigmoid(),
        )

        self.E_MLP_g = nn.Sequential(
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="relu"),
            nn.GELU(),
            Linear(num_hidden, num_hidden, init="gating"),
            nn.Sigmoid(),
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        if self.node_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_V = h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])

        if self.edge_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_E = h_E * self.E_MLP_g(c_V[batch_id[edge_idx[0]]])

        return h_V, h_E


class GeneralGNN(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads=None,
        scale=30,
        node_net="AttMLP",
        edge_net="EdgeMLP",
        node_context=0,
        edge_context=0,
        norm_class: nn.Module = nn.BatchNorm1d,
    ):
        super(GeneralGNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.norm_class = norm_class
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([self.norm_class(num_hidden) for _ in range(3)])
        self.node_net = node_net
        self.edge_net = edge_net
        if node_net == "AttMLP":
            self.attention = NeighborAttention(num_hidden, num_in, num_heads=4)
        if edge_net == "None":
            pass
        if edge_net == "EdgeMLP":
            self.edge_update = EdgeMLP(
                num_hidden, num_in, num_heads=4, norm_class=norm_class
            )

        self.context = Context(
            num_hidden,
            num_in,
            num_heads=4,
            node_context=node_context,
            edge_context=edge_context,
        )

        self.dense = nn.Sequential(
            Linear(num_hidden, num_hidden * 4, init="relu"),
            nn.GELU(),
            Linear(num_hidden * 4, num_hidden, init="final"),
        )
        # TODO: consider removing as don't do anything?
        self.W11 = Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = Linear(num_hidden, num_hidden, bias=True)
        self.W13 = Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(
        self, h_V: th.Tensor, h_E: th.Tensor, edge_idx: th.Tensor, batch_id: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """Inputs:
        * h_V: (N, d1), node features
        * h_E: (E, d2), edge features
        * edge_idx: (2, E), edge indices
        * batch_id: (N), node batch ids
        """
        src_idx, dst_idx = edge_idx
        # TODO: hypnopump@ consider pre-norm blocks instead of post-norm
        if self.node_net == "AttMLP" or self.node_net == "QKV":
            dh = self.attention(
                h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx, batch_id, dst_idx
            )
        else:
            dh = self.attention(h_V, h_E, src_idx, batch_id, dst_idx)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if self.edge_net == "None":
            pass
        else:
            h_E = self.edge_update(h_V, h_E, edge_idx, batch_id)

        h_V, h_E = self.context(h_V, h_E, edge_idx, batch_id)
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        dropout: float = 0,
        node_net: str = "AttMLP",
        edge_net: str = "EdgeMLP",
        node_context: bool = True,
        edge_context: bool = False,
        checkpoint: bool = False,
        norm_class: nn.Module = nn.BatchNorm1d,
    ):
        """Graph labeling network"""
        super(StructureEncoder, self).__init__()
        self.checkpoint = checkpoint

        encoder_layers = []
        module = GeneralGNN
        for i in range(num_encoder_layers):
            encoder_layers.append(
                module(
                    hidden_dim,
                    hidden_dim * 2,
                    dropout=dropout,
                    node_net=node_net,
                    edge_net=edge_net,
                    node_context=node_context,
                    edge_context=edge_context,
                    norm_class=norm_class,
                ),
            )

        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, h_V, h_P, P_idx, batch_id):
        for layer in self.encoder_layers:
            if self.checkpoint:
                h_V, h_P = checkpoint.checkpoint(layer, h_V, h_P, P_idx, batch_id)
            else:
                h_V, h_P = layer(h_V, h_P, P_idx, batch_id)
        return h_V, h_P


class MLPDecoder(nn.Module):
    def __init__(
        self, hidden_dim: int, vocab: int = 20, norm_class: nn.Module = nn.BatchNorm1d
    ):
        super().__init__()
        self.vocab = vocab
        # TODO: hypnopump@ consider potential bottleneck here: norm+mlp?
        self.readout = Linear(hidden_dim, vocab, init="final")

    def forward(self, h_V, batch_id=None):
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits


if __name__ == "__main__":
    pass
