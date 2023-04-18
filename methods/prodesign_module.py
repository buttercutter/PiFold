from typing import Optional

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

from utils import batched_index_select, gather_nodes

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
    # Mask of first-order neighboring nodes: 1 indicates the node exists, 0 indicates the node does not exist.
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1)
    # The mask of self * the mask of neighboring nodes
    mask_attend = mask.unsqueeze(-1) * mask_attend
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

    def dense_fw(
        self,
        h_V: th.Tensor,
        h_E: th.Tensor,
        edge_idx: th.Tensor,
        node_mask: th.Tensor,
        edge_mask: th.Tensor,
    ) -> th.Tensor:
        """Analog for dense inputs.
        Inputs:
        * h_V: (b, n, d) node features
        * h_E: (b, n, k, dE) edge features
        * edge_idx: (b, n, k) edge indices
        * node_mask: (b, n) bool mask
        * edge_mask: (b, n, k) bool mask
        Outputs: (b, n, d)
        """
        h = self.num_heads
        d = int(self.num_hidden / self.num_heads)
        b, n, k, dE = h_E.shape
        max_neg = -torch.finfo(h_E.dtype).max

        h_V_k2n = h_V[..., None, :].repeat(1, 1, k, 1)  # (b, n, k, d)
        V = self.W_V(h_E).reshape(b, n, k, h, d)  # (b, n, k, h, d_head)
        w = self.Bias(th.cat((h_V_k2n, h_E), dim=-1))  # (b, n, k, h)
        w = w.mul_(1 / d ** 0.5)
        w.masked_fill_(~edge_mask.unsqueeze(-1), max_neg)

        attend = w.softmax(dim=-2)[..., None]  # (b, n, k, h, ())
        h_V = (attend * V).sum(dim=-3).view(b, n, -1) # (b, n, d)

        h_V_update = h_V
        if self.output_mlp:
            h_V_update = self.W_O(h_V)

        return h_V_update

    def forward(
        self,
        h_V: th.Tensor,
        h_E: th.Tensor,
        center_id: th.Tensor,
        batch_id: th.Tensor,
        dst_idx: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        # (N,d1), (E,d2) -> (E,d1), (E,d2) -> (E,d1+d2) -> (E, heads, 1)
        w = self.Bias(th.cat([h_V[center_id], h_E], dim=-1)).mul_(1/d**0.5)
        w = w.view(E, n_heads, 1)
        # (E, d2) -> (E, heads, d)
        V = self.W_V(h_E).view(-1, n_heads, d)

        # (E, heads, 1)
        attend = scatter_softmax(w, index=center_id, dim=0)
        # (E, heads, d) * (E, heads, 1) -> (N, heads * d)
        h_V = scatter_sum(attend * V, center_id, dim=0).view(h_V.shape[0], -1)

        h_V_update = h_V
        if self.output_mlp:
            h_V_update = self.W_O(h_V)

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
        train_mode: str = "sparse",
    ):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = norm_class(num_hidden)
        self.train_mode = train_mode
        # TODO: hypnopump@: if not loading published models, turn into MLP ???
        self.W11 = Linear(num_hidden + num_in, num_hidden, bias=True, init="relu")
        self.W12 = Linear(num_hidden, num_hidden, bias=True, init="relu")
        self.W13 = Linear(num_hidden, num_hidden, bias=True, init="final")
        self.act = torch.nn.GELU()

    def forward(
        self,
        h_V,
        h_E,
        edge_idx,
        batch_id: th.Tensor,
        node_mask: Optional[th.Tensor] = None,
        edge_mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        if self.train_mode == "dense":
            # (b, n, d), (b, n, k, dE), (b, n, d) -> (b, n, k, d), (b, n, k, dE), (b, n, (), d) -> (b, n, k, d+dE+d)
            h_V_k2n = batched_index_select(h_V, edge_idx, dim=-2)
            h_V_n2k = h_V[..., None, :].expand_as(h_V_k2n)
            h_EV = th.cat((h_V_k2n, h_E, h_V_n2k), dim=-1)

        elif self.train_mode == "sparse":
            # (N, d1), (E, d2), (N, d1) -> (E, d1+d2+d1)
            src_idx, dst_idx = edge_idx
            h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)

        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        # TODO: hypnopump@: if inference and memory consumption is a problem, use inplace operation
        # TODO: hypnopump@: clarify reisudal better ??? either inside-block or outside-block, not mixed
        h_E = self.norm(h_E + self.dropout(h_message), mask=edge_mask)
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
        node_context: bool=False,
        edge_context: bool=False,
        train_mode: str = "sparse",
    ):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context
        self.train_mode = train_mode

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

    def forward(
        self,
        h_V,
        h_E,
        edge_idx,
        batch_id: th.Tensor,
        node_mask: Optional[th.Tensor] = None,
        edge_mask: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        if self.train_mode == "dense":
            if self.node_context or self.edge_context:
                float_mask = node_mask.to(h_V.dtype)[..., None]
                num_nodes = float_mask.sum(dim=-2).add_(1e-5)
                # (B N C) -> (B C)
                h_V_mean = (h_V * float_mask).sum(dim=-2) / num_nodes

            if self.node_context:  # (b n c) * (b () c)
                h_V = h_V * self.V_MLP_g(h_V_mean)[..., None, :]
            if self.edge_context:  # (b n c) * (b () c)
                h_E = h_E * self.E_MLP_g(h_V_mean)[..., None, :]

            return h_V, h_E

        if self.node_context or self.edge_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)

        if self.node_context:
            h_V = h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])

        if self.edge_context:
            h_E = h_E * self.E_MLP_g(c_V[batch_id[edge_idx[0]]])

        return h_V, h_E


class GeneralGNN(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads: int = 1,
        scale=30,
        node_net="AttMLP",
        edge_net="EdgeMLP",
        node_context=0,
        edge_context=0,
        norm_class: nn.Module = nn.BatchNorm1d,
        train_mode: str = "sparse",
    ):
        super(GeneralGNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([norm_class(num_hidden) for _ in range(2)])
        self.node_net = node_net
        self.edge_net = edge_net
        self.train_mode = train_mode
        if node_net == "AttMLP":
            self.attention = NeighborAttention(num_hidden, num_in, num_heads=num_heads)
        if edge_net == "None":
            pass
        if edge_net == "EdgeMLP":
            self.edge_update = EdgeMLP(
                num_hidden, num_in, num_heads=num_heads, norm_class=norm_class, train_mode=train_mode
            )

        self.context = Context(
            num_hidden,
            num_in,
            num_heads=num_heads,
            node_context=node_context,
            edge_context=edge_context,
            train_mode=train_mode,
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
        self,
        h_V: th.Tensor,
        h_E: th.Tensor,
        edge_idx: Optional[th.Tensor] = None,
        batch_id: Optional[th.Tensor] = None,
        node_mask: Optional[th.Tensor] = None,
        edge_mask: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Implements the `sparse` forward. See `self.dense_forward` for dense.
        Inputs:
        * h_V: (N, d1), node features
        * h_E: (E, d2), edge features
        * edge_idx: (2, E), edge indices
        * batch_id: (N), node batch ids
        * mode: "sparse" or "dense"
        * node_mask: (B, N), node mask. only required for dense mode
        * edge_mask: (B, N, K), edge mask. only required for dense mode
        """
        # sparse method
        if self.train_mode == "dense":
            if self.node_net == "AttMLP" or self.node_net == "QKV":
                h_EV = batched_index_select(h_V, edge_idx, dim=-2)
                h_EV = torch.cat([h_E, h_EV], dim=-1)
                # h_EV = torch.cat([h_E, h_V[..., None, :].expand_as(h_E)], dim=-1)
                dh = self.attention.dense_fw(h_V, h_EV, edge_idx, node_mask, edge_mask)
            else:
                dh = self.attention.dense_fw(h_V, h_E, edge_idx, node_mask, edge_mask)

        if self.train_mode == "sparse":
            src_idx, dst_idx = edge_idx

            if self.node_net == "AttMLP" or self.node_net == "QKV":
                h_EV = torch.cat([h_E, h_V[dst_idx]], dim=-1)
                dh = self.attention(h_V, h_EV, src_idx, batch_id, dst_idx)
            else:
                dh = self.attention(h_V, h_E, src_idx, batch_id, dst_idx)

        h_V = self.norm[0](h_V + self.dropout(dh), mask=node_mask)
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh), mask=node_mask)

        # FIXME: hypnopump@ residual should be here and keep a clean path (norm on addition)
        if self.edge_net == "None":
            pass
        else:
            h_E = self.edge_update(
                h_V,
                h_E,
                edge_idx,
                batch_id,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )

        h_V, h_E = self.context(
            h_V,
            h_E,
            edge_idx,
            batch_id,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        dropout: float = 0,
        num_heads: int = 1,
        node_net: str = "AttMLP",
        edge_net: str = "EdgeMLP",
        node_context: bool = True,
        edge_context: bool = False,
        checkpoint: bool = False,
        norm_class: nn.Module = nn.BatchNorm1d,
        train_mode: str = "sparse",
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
                    num_heads=num_heads,
                    node_net=node_net,
                    edge_net=edge_net,
                    node_context=node_context,
                    edge_context=edge_context,
                    norm_class=norm_class,
                    train_mode=train_mode,
                ),
            )

        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(
        self,
        h_V,
        h_P,
        P_idx,
        batch_id,
        node_mask: Optional[th.Tensor] = None,
        edge_mask: Optional[th.Tensor] = None,
    ):
        """Inputs:
        ...
        * mode: str, "sparse" or "dense"
        * node_mask: (B, N), node mask. required if mode == "dense"
        * edge_mask: (B, E), edge mask. required if mode == "dense"
        """
        for layer in self.encoder_layers:
            if self.checkpoint:
                h_V, h_P = checkpoint.checkpoint(
                    layer,
                    h_V,
                    h_P,
                    P_idx,
                    batch_id,
                    node_mask,
                    edge_mask,
                )
            else:
                h_V, h_P = layer(
                    h_V,
                    h_P,
                    P_idx,
                    batch_id,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                )
        return h_V, h_P


class MLPDecoder(nn.Module):
    def __init__(
        self, hidden_dim, vocab: int = 20, norm_class: nn.Module = nn.BatchNorm1d, train_mode: str = "sparse",
    ):
        super().__init__()
        self.readout = Linear(hidden_dim, vocab, init="final")
        self.norm_class = norm_class
        self.train_mode = train_mode

    def forward(
        self,
        h_V: th.Tensor,
        batch_id: Optional[th.Tensor] = None,
        node_mask: Optional[th.Tensor] = None,
        edge_mask: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits


if __name__ == "__main__":
    pass
