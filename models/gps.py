from typing import Optional
from models.encoder import Global3DCentralityEncodingSparse, GlobalEncoder, CentEncoder, NodeEncoder, EdgeEncoder, MLPEncoder, SPDAttentionBias, LaplacianPEVal, LaplacianPEVec, RandomWalkStructuralEncoding, RingEncoder, GlobalLaplacianPEVal
from models.distance import GaussianDistanceEmbedding
from models.lossfunc import FocalLoss, MultiCenterSVDDLoss, AsymmetricLoss, CenterLoss
import torch
from torch import nn
from torch.nn import functional as F
import math
import configs.parameters as cf
from torch.amp import autocast
from torch_scatter import scatter_min
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter
from torch_geometric.utils import to_dense_batch
from utils.helpers import build_dense_batch, pad_to_max_nodes
from torch_geometric.utils import softmax        # segment-wise softmax
from torch_scatter import scatter_add            # efficient segmented sum
    

def finite(name, t):
    mask = ~torch.isfinite(t)
    if mask.any():
        # coords of first few non-finite entries
        bad_coords = mask.nonzero(as_tuple=False)  # shape (N, ndim)
        # the actual values at those positions
        bad_vals   = t[mask]                       # shape (N,)
        # take up to 5
        coords = bad_coords[:5].tolist()
        vals   = bad_vals[:5].tolist()
        raise RuntimeError(
            f"{name} has non-finite values at {coords} with values {vals}"
        )

#####################################################################
# Utility functions #################################################
#####################################################################

def _init_linear(in_dim: int, out_dim: int, bias: bool = True):
    """Xavier‐initialised linear layer used throughout the model."""
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(lin.weight)
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


def _mlp(channels, act=nn.GELU, bias: bool = True):
    """Two‑layer MLP as used in the paper (Eq. 12‑14 & 16‑18)."""
    return nn.Sequential(
        _init_linear(channels[0], channels[1], bias),
        act(),
        nn.LayerNorm(channels[1]),
        _init_linear(channels[1], channels[2], bias),
    )

#####################################################################
# Message‑Passing layer #############################################
#####################################################################

class GraphDropout(nn.Module):
    """
    Dropout at the granularity of entire graphs (i.e. batch elements).
    With probability `p` each graph’s node features are zeroed out;
    otherwise they are scaled by 1/(1-p) to preserve expectation.
    """
    def __init__(self, p: float):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability has to be in [0,1)")
        self.p = p

    def forward(self, x: torch.Tensor, batch = None) -> torch.Tensor:
        """
        x: [batch, num_nodes, dim_node]
        """
        # No dropout during evaluation or if drop prob is 0
        if not self.training or self.p == 0.0:
            return x

        keep_rate = 1.0 - self.p

        # 1) Dense case: [B, N, D]
        if x.dim() == 3 and batch is None:
            B, _, _ = x.shape
            # one mask per graph
            mask = torch.bernoulli(
                torch.full((B, 1, 1), keep_rate, 
                           device=x.device, dtype=x.dtype)
            )
            return x.div(keep_rate).mul(mask)

        # 2) Flat case: [N_tot, D] + batch vector
        elif x.dim() == 2 and batch is not None:
            # determine how many graphs
            num_graphs = int(batch.max().item()) + 1
            # one mask per graph
            mask_per_graph = torch.bernoulli(
                torch.full((num_graphs, 1), keep_rate, 
                           device=x.device, dtype=x.dtype)
            )
            # expand to each node
            node_mask = mask_per_graph[batch]  # [N_tot, 1]
            return x.div(keep_rate).mul(node_mask)

        else:
            raise ValueError(
                "GraphDropout got x of shape %s with batch=%s; "
                "expected either (3D, batch=None) or (2D, batch=LongTensor)" 
                % (tuple(x.shape), batch)
            )


class MPNNLayer(nn.Module):
    """GPS++ message‑passing update (Eq. 5‑11).

    This implementation closely follows the pseudocode in the paper but
    uses torch_scatter for efficiency.  Edge and global states are **optional**
    so the layer can be used in ablation studies.
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_face: int, dim_global: int, use_face: bool, dropout_edge: float = 0.0035,
    # def __init__(self, dim_node: int, dim_edge: int, dim_face: int, dim_global: int, dropout_edge: float = 0.0035,
        dropout_node: float = 0.3, dropout_global: float = 0.35, face_alpha: float = 1.0): 
        super().__init__()
        self.use_face = use_face
        self.face_alpha = float(face_alpha) 
        face_extra = (dim_face + dim_edge) if use_face else 0
        face_global = dim_face if use_face else 0
        self.emb_face = dim_edge + dim_face + dim_global
        self.emb_edge = 2*dim_node + dim_edge + dim_global + face_extra
        self.emb_node = dim_node + 2*(dim_edge + dim_node) + dim_global
        self.emb_global = dim_global + dim_node + dim_edge + face_global
        self.mlp_face = _mlp([self.emb_face, dim_face*4, dim_face])
        self.mlp_edge = _mlp([self.emb_edge, dim_edge*4, dim_edge])
        self.mlp_node = _mlp([self.emb_node, dim_node*4, dim_node])
        self.mlp_global = _mlp([self.emb_global, dim_global*4, dim_global])

        self.drop_e = dropout_edge
        self.drop_n = dropout_node
        self.drop_g = dropout_global

        # layer‐norm on the node‐update output
        self.norm = nn.LayerNorm(dim_node)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_attr: torch.Tensor,
                face_index: torch.LongTensor,
                face_attr: torch.Tensor,
                g: torch.Tensor,
                node_batch):
        """
        x          [N, dim_node]       node features at layer ℓ
        edge_index [2, E]              the usual G-Geom edge index
        edge_attr  [E, dim_edge]       edge features at layer ℓ
        face_index [2, Fxe_f]          column lengths equal to number of faces times number of bonds in each face
        face_attr  [F, dim_face]       face features at layer
        g          [dim_global]        global feature at layer ℓ
        node_batch [N]                 indicating which batch each node belongs to
        no_faces   [B]                 indicating how many faces for each graph
        → returns (x_new, edge_attr_new, g_new)
        """
            
        row, col = edge_index  # row = senders, col = receivers
        B = g.size(0)
        edge_batch = node_batch[row] 

        if self.use_face:
        # if cf.FACE:
            total_faces = face_attr.size(0)
            face_idx, bond_idx = face_index
            face_batch = edge_batch[bond_idx] 
            positions = torch.arange(face_idx.size(0), device=face_idx.device)

            # 2) for each face, find the min position (i.e. first occurrence)
            #    scatter_min returns a tuple (min_values, argmin_indices)
            _, first_pos = scatter_min(
                positions, face_idx,
                dim=0,
                dim_size=total_faces
            )
            face_batch_per_face = face_batch[first_pos]
            g_f = g[face_batch_per_face]  # [N_F, 1]
            edge_inc = edge_attr[bond_idx]

            sum_edges_per_face = scatter(edge_inc, face_idx, dim=0, dim_size=total_faces, reduce="sum") # [N_F, face_dim]
            face_input = torch.cat([sum_edges_per_face, face_attr, g_f], dim=-1)
            face_bar = self.mlp_face(face_input)

            deg_face = scatter(torch.ones_like(face_idx, dtype=edge_inc.dtype),
                   face_idx, dim=0, dim_size=total_faces, reduce="sum"
                  ).clamp_min(1.0).unsqueeze(-1)
            deg_minus1 = (deg_face[face_idx] - 1.0).clamp_min(1.0)

            # create a set for each face where it is the sum of all edges excluding one edge
            sum_edges_exclude_self = sum_edges_per_face[face_idx] - edge_inc
            mean_except_self = (sum_edges_exclude_self) / deg_minus1
            face_message = torch.cat([face_bar[face_idx], mean_except_self], dim=-1)

            # Aggregate across all faces containing each edge
            coface_sum_per_edge = scatter(face_message, bond_idx, dim=0,
                                        dim_size=edge_attr.size(0), reduce="sum")
            coface_sum_per_edge = self.face_alpha * coface_sum_per_edge

        # 1) Edge update: ē_{uv} = Dropout( MLP_edge[x_u | x_v | e_uv | g] )
        # repeat global to match #edges
        g_e = g[edge_batch] 
        if self.use_face:
        # if cf.FACE:
            edge_input = torch.cat([x[row], x[col], edge_attr, coface_sum_per_edge, g_e], dim=-1) 
        else:
            edge_input = torch.cat([x[row], x[col], edge_attr, g_e], dim=-1)
        e_bar = F.dropout(self.mlp_edge(edge_input),
                          p=self.drop_e,
                          training=self.training)

        # 2) Node update:
        #    sum_in  = Σ_u [ ē_{u→i} | x_u ]
        #    sum_out = Σ_v [ ē_{i→v} | x_v ]
        send_msg = torch.cat([e_bar, x[row]], dim=-1)
        sum_in  = scatter(send_msg, col, dim=0,
                          dim_size=x.size(0), reduce="sum")
        recv_msg = torch.cat([e_bar, x[col]], dim=-1)
        sum_out = scatter(recv_msg, row, dim=0,
                          dim_size=x.size(0), reduce="sum")

        # build node MLP input and run
        g_n = g[node_batch]
        node_input = torch.cat([x, sum_in, sum_out, g_n], dim=-1)
        x_bar = self.mlp_node(node_input)

        # 3) Global update: ḡ = MLP_global[ g | Σ_j x̄_j | Σ_{uv} ē_{uv} ]
        # sum_x = x_bar.sum(dim=0, keepdim=True)      # [1, dim_node]
        # sum_e = e_bar.sum(dim=0, keepdim=True)      # [1, dim_edge? out_dim_edge?]
        sum_x = scatter(x_bar,     # [N, dim_node]
                node_batch,
                dim=0,
                dim_size=B,
                reduce="sum")

        # sum ē per graph (→ [B, dim_edge])
        sum_e = scatter(e_bar,     # [E, dim_edge]
                        edge_batch,
                        dim=0,
                        dim_size=B,
                        reduce="sum")
        if self.use_face:
        # if cf.FACE:
            sum_f = scatter(face_bar, face_batch_per_face, dim=0, dim_size=B, reduce="sum")
            g_input = torch.cat([g, sum_x, sum_e, sum_f], dim=-1)
        else:
            g_input = torch.cat([g, sum_x, sum_e], dim=-1) 
        g_bar = self.mlp_global(g_input)

        # 4) Residual + norm on nodes
        x = self.norm(F.dropout(x_bar, p=self.drop_n, training=self.training)) + x

        # 5) Residual on edges & global
        edge_attr = edge_attr + e_bar
        if self.use_face:
        # if cf.FACE:
            face_attr = face_attr + face_bar
        g         = g + F.dropout(g_bar, p=self.drop_g, training=self.training)

        return x, edge_attr, face_attr, g

    def set_face_alpha(self, alpha: float):
        self.face_alpha = float(alpha)


#####################################################################
# Biased multi‑head self‑attention ##################################
#####################################################################
class BiasedSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_attn_heads, attention_dropout_rate = 0.3, output_dropout_rate = 0, graph_dropout_rate = 0.3, prediction_head=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_attn_heads = n_attn_heads
        self.head_size = hidden_dim // n_attn_heads
        self.scale = (self.head_size ** -0.5)

        # initialise weight range just like TF’s RandomUniform(-hidden_dim**-0.5, +...)
        init_rad = hidden_dim ** -0.5
        for name in ('q','k','v','proj') if not prediction_head else ('q','k','v',):
            setattr(self, name,
                    nn.Linear(hidden_dim, hidden_dim, bias=True))
            # override default init
            nn.init.uniform_(getattr(self, name).weight, -init_rad, init_rad)
            nn.init.uniform_(getattr(self, name).bias,   -init_rad, init_rad)

        self.prediction_head = prediction_head
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.out_dropout = nn.Dropout(output_dropout_rate)
        self.graph_dropout = GraphDropout(graph_dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)


    def forward(self, x, batch, attn_bias):
        # 1) pack nodes into a dense [B, Nmax, d_node] tensor, get mask [B, Nmax]
        x_dense, mask = to_dense_batch(x, batch)
        b, n, _  = x_dense.shape
        # x_dense: [B, Nmax, d_node],  mask: BoolTensor [B, Nmax]

        # project
        q = self.q(x_dense).view(b, n, self.n_attn_heads, self.head_size)
        k = self.k(x_dense).view(b, n, self.n_attn_heads, self.head_size)
        v = self.v(x_dense).view(b, n, self.n_attn_heads, self.head_size)

        # reshape for matmul
        q = q.permute(0,2,1,3)            # [b, heads, nodes, head_size]
        k = k.permute(0,2,3,1)            # [b, heads, head_size, nodes]
        v = v.permute(0,2,1,3)            # [b, heads, nodes, head_size]

        mask2 = mask.unsqueeze(1).unsqueeze(-1)  # [B,1,Nmax,1]
        mask3 = mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,Nmax]
        valid_pairs = mask2 & mask3 
        # scaled dot-product + bias
        attn = torch.matmul(q, k) * self.scale

        if getattr(self, "debug_print_attn", False) and (not self.training):
            with torch.no_grad():
                li = -1 if self._layer_idx is None else self._layer_idx
                vp = valid_pairs.expand_as(attn)  # [B,H,N,N]
                def _p(name, t):
                    vals = t[vp]
                    if vals.numel():
                        print(f"[attn dbg][L{li}] {name}: "
                            f"min={vals.amin().item():.4f} "
                            f"max={vals.amax().item():.4f} "
                            f"mean={vals.mean().item():.4f} "
                            f"std={vals.std(unbiased=False).item():.4f}")
                _p("logits", attn)
                if attn_bias is not None:
                    _p("attn_bias", attn_bias)
                    _p("logits_plus_bias", attn + attn_bias)

        if attn_bias is not None: 
            attn = attn + attn_bias          # [b, heads, nodes, nodes]
        # 1) Exclude invalid KEYS with -inf (safe)
        attn = attn.masked_fill(~mask3, float("-inf"))

        # 2) Avoid all -inf rows for invalid QUERIES: set to 0 (or any const)
        attn = attn.masked_fill(~mask2, 0.0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 3) Ensure no contribution from padded QUERIES
        attn = attn.masked_fill(~mask2, 0.0)

        # ---- LOGGING (eval-only; optional) ----
        if getattr(self, "_log_attn", False) and self.training is False:
            # assume the model stored a ref to the shared SPD bias module
            spd_dense = None
            if hasattr(self, "_get_spd_dense"):
                spd_dense, _mask_from_bias = self._get_spd_dense()
            if (spd_dense is not None) and hasattr(self, "_mlogger"):
                # use the same mask from to_dense_batch (more reliable)
                self._mlogger.update(self._layer_idx, attn.detach().cpu(),
                                     mask.detach().cpu(), spd_dense.detach().cpu())

        if self.prediction_head:
            # return attention weights and the values
            return attn, v

        # attend to values
        out = torch.matmul(attn, v)      # [b, heads, nodes, head_size]
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(b, n, self.hidden_dim)  # [b, nodes, hidden_dim]

        # final linear projection
        out = self.proj(out)
        out = self.out_dropout(out)
        
        out = self.graph_dropout(out)
        out = out[mask]
        out = out + x
        out = self.layer_norm(out)

        return out


class GPSFeedForward(nn.Module):
    def __init__(
        self,
        d_node: int,
        hidden_mult: int = 4,
        dropout_p: float = 0,
        graph_dropout_p: float = 0.3,
    ):
        """
        Position‐wise FFN for GPS++ (Eq. …).
        
        y = FFN(x) computed as
          \bar x = Dropout_p( GELU( Dense1(x) ) )    ∈ R^{N_tot × (hidden_mult·d_node)}
          y      = GraphDropout_{graph_dropout_p}( Dense2(\bar x) ) + x  ∈ R^{N_tot × d_node}

        Args:
          d_node:           input/output dimension of node features.
          hidden_mult:      inner dimension is hidden_mult * d_node.
          dropout_p:        p for dropout on the hidden layer.
          graph_dropout_p:  p for dropout on the output layer before residual.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_node, hidden_mult * d_node)
        self.fc2 = nn.Linear(hidden_mult * d_node, d_node)
        self.dropout_p = dropout_p
        self.graph_dropout_p = GraphDropout(graph_dropout_p)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_node)

    def forward(self, x, batch):
        """
        data: a torch_geometric.data.Batch or Data with
            - data.x: FloatTensor of shape [N_tot, d_node]

        returns:
            y: FloatTensor of shape [N_tot, d_node]
        """
        h = self.fc1(x)                                  # [N_tot, hidden_mult*d_node]
        h = self.act(h)                                    # nonlinearity
        h = F.dropout(h, p=self.dropout_p, training=self.training)
                                                        # dropout on hidden
        h = self.fc2(h)                                  # [N_tot, d_node]
        # x_dense, mask = to_dense_batch(h, batch)
        h = self.graph_dropout_p(h, batch)
        # h = h[mask]
        h = h + x
        h = self.layer_norm(h)
                                                        # “graph‐dropout” on output
        return h                                    # residual skip


#####################################################################
# GPS++ block #######################################################
#####################################################################

class GPSBlock(nn.Module):
    """One complete GPS++ layer: MPNN ‖ Attention → sum → FFN (Eq. 1‑4)."""

    def __init__(self, dim_node: int, dim_edge: int, dim_face: int, dim_global: int, cfg,
                 dropout_edge: float = 0.0035, dropout_node: float = 0.3, dropout_global: float = 0.35):
        super().__init__()
        self.cfg = cfg
        self.mpn = MPNNLayer(dim_node, dim_edge, dim_face, dim_global, cfg.face, dropout_edge, dropout_node, dropout_global)
        if cfg.attention:
            self.attn = BiasedSelfAttentionLayer(dim_node, cfg.num_heads)
            self.fnn = GPSFeedForward(dim_node)

    def forward(self, x, edge_index, edge_attr, face_index, face_attr, global_state, edge_batch, attn_bias=None):
        # Parallel paths
        x_mpn, edge_attr, face_attr, global_state = self.mpn(x, edge_index, edge_attr, face_index, face_attr, global_state, edge_batch)
        if self.cfg.attention:
            x_attn = self.attn(x, edge_batch, attn_bias)

            # Combine (Eq. 3) and feed‑forward (Eq. 4).
            x = x_mpn + x_attn
            x = self.fnn(x, edge_batch)
        else:
            x = x_mpn
        return x, edge_attr, face_attr, global_state


#####################################################################
# Encoder & Decoder #################################################
#####################################################################

class OutputDecoder(nn.Module):
    """Two‑layer MLP after global pooling (Section 4.2)."""

    def __init__(self, input_dim: int, num_labels=3):
        super().__init__()
        self.mlp = _mlp([input_dim, input_dim*4, num_labels])

    def forward(self, x):
        return self.mlp(x)

class SVDDDecoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim=64):
        super().__init__()
        self.mlp = _mlp([input_dim, input_dim*4, embed_dim], bias=True)
    
    def forward(self, x):
        return self.mlp(x)

class FastAttentionPool(nn.Module):
    """
    Memory-efficient global attention pooling.
    Scores nodes with a single learnable vector; optional target conditioning.
    """
    def __init__(self, d_node: int, d_target: int = 0):
        super().__init__()
        self.w_x  = nn.Parameter(torch.empty(d_node))   # node scorer
        nn.init.normal_(self.w_x, std=0.02)
        self.bias = nn.Parameter(torch.zeros(1))
        self.W_t  = nn.Linear(d_target, 1, bias=False) if d_target > 0 else None

    @staticmethod
    def _num_graphs(batch: torch.Tensor) -> int:
        return int(batch.max().item()) + 1 if batch.numel() else 0

    def forward(self,
                x: torch.Tensor,             # [∑N, d_node]
                batch: torch.Tensor,         # [∑N]
                target_embed: torch.Tensor | None = None  # [B, d_target] or None
               ) -> torch.Tensor:            # -> [B, d_node]
        # 1) Cheap per-node score
        logits = x @ self.w_x + self.bias.squeeze(0)    # [∑N]

        # 2) Optional target term, broadcast per node
        if self.W_t is not None and target_embed is not None:
            logits = logits + self.W_t(target_embed).squeeze(-1)[batch]  # [∑N]

        # 3) Segment-wise softmax (normalized within each graph)
        attn = softmax(logits, batch)                    # [∑N], sums to 1 per graph

        # 4) Weighted sum per graph (no [B, N, d] tensors)
        x_weighted = x * attn.unsqueeze(-1)              # [∑N, d_node]
        B = self._num_graphs(batch)
        out = x.new_zeros(B, x.size(-1))
        out = scatter_add(x_weighted, batch, dim=0, out=out)  # [B, d_node]
        return out

#####################################################################
# The full GPS++ network ###########################################
#####################################################################

class GPSPlusPlus(nn.Module):
    """PyTorch implementation of the GPS++ architecture.

    This version is **framework‑agnostic** (standard PyTorch + PyG) so it
    can be trained on CPU/GPU instead of Graphcore IPUs.  It purposefully
    mirrors the layer counts and hyper‑parameters from the original paper
    but exposes them as arguments for easy experimentation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        face_dim: int,
        global_dim: int,
        # num_heads: int,
        node_chem_dict: dict,
        edge_chem_dict: dict,
        cfg,
        max_degree: int,
        face_chem_dict: dict = None, 
        dropout_edge: float = 0.0035,
        dropout_node: float = 0.3, 
        dropout_global: float = 0.35,
        kernel_number: int = 128,
        pos_weight = [1, 1, 1]
    ):
        super().__init__()
        self.cfg = cfg
        self.num_layers = cfg.num_layers
        self.layers = nn.ModuleList([
            GPSBlock(node_dim, edge_dim, face_dim, global_dim, cfg, dropout_edge, dropout_node, dropout_global) for _ in range(self.num_layers)
        ])
        self.output_type = ['infer', 'loss']
        self.loss_type = cfg.loss
        self.position = cfg.position
        self.attention = cfg.attention
        # --- Probes / toggles ---
        self.eval_face_noise_sigma: float = getattr(cfg, "eval_face_noise_sigma", 0.0)
        self.enable_face_noise_only_eval: bool = True  # only apply in eval mode
        self.node_chem_embedder = NodeEncoder(**node_chem_dict)
        self.edge_chem_embedder = EdgeEncoder(**edge_chem_dict)
        self.node_chem_encoder = _mlp([64, node_dim*4, node_dim])
        self.edge_chem_encoder = _mlp([64, edge_dim*4, edge_dim])
        self.xlocal_encoder = CentEncoder(max_degree, 64)
        if self.cfg.barycentric:
            self.node_laplace_vec = LaplacianPEVec()
            self.edge_laplace_vec = LaplacianPEVec()
            self.ring_laplace_vec = LaplacianPEVec()
            self.global_laplace_val = GlobalLaplacianPEVal(global_dim=global_dim)
            self.node_rw_encoder = RandomWalkStructuralEncoding()
            self.edge_rw_encoder = RandomWalkStructuralEncoding()
            self.ring_rw_encoder = RandomWalkStructuralEncoding()
        else:
            self.laplace_val = LaplacianPEVal()
            self.laplace_vec = LaplacianPEVec()
            self.random_walk_encoder = RandomWalkStructuralEncoding()
        if self.cfg.face:
            self.face_chem_embedder = RingEncoder(**face_chem_dict)
            self.face_chem_encoder = _mlp([64, face_dim*4, face_dim])
        if self.position:
            self.distance_embedder = GaussianDistanceEmbedding(kernel_number)
            self.bl_encoder = MLPEncoder(kernel_number, 2, 32)
            self.x3d_encoder = Global3DCentralityEncodingSparse(kernel_number, 32)
            self.project_x = nn.Linear(node_dim+64+32+64+32, node_dim)
            self.project_edge = nn.Linear(edge_dim+32, edge_dim)
        else:
            if self.cfg.barycentric:
                self.project_x = nn.Linear(node_dim+32+32+64, node_dim)
                self.project_edge = nn.Linear(edge_dim+32+32, edge_dim)
                self.project_face = nn.Linear(face_dim+32+32, face_dim)
            else:
                self.project_x = nn.Linear(node_dim+64+32+64, node_dim)
                self.project_edge = nn.Linear(edge_dim, edge_dim)          
        self.global_encoder = GlobalEncoder()
        # self.pool = FastAttentionPool(d_node=node_dim, d_target=0)
        self.decoder = OutputDecoder(node_dim)
        if self.loss_type == "FOCAL_SVDD" or self.loss_type == "ASYM_SVDD" or self.loss_type == "FOCAL_CENTER" or self.loss_type == "ASYM_CENTER":
            self.svdd_decoder = SVDDDecoder(node_dim)

        if self.attention:
            max_distance = 55 if cfg.barycentric else 35
            self.spd_embedder = SPDAttentionBias(cfg.num_heads, max_distance)
            from utils.attn_logging import MultiLayerAttnLogger
            self._log_attention = False
            self._attn_logger = MultiLayerAttnLogger(n_layers=self.num_layers)
            self._attn_logger.set_overflow_bucket(self.spd_embedder.overflow_bucket)

            get_spd_dense = self.spd_embedder.get_last_spd_dense  # function, not nn.Module
            for li, block in enumerate(self.layers):
                if hasattr(block, "attn"):
                    block.attn._get_spd_dense = get_spd_dense
                    block.attn._mlogger = self._attn_logger
                    block.attn._layer_idx = li
                    block.attn._log_attn = False
        # (Optional) learnable global vector as in many GPS variants
        # self.global_token = nn.Parameter(torch.zeros(1, hidden_dim))
        # nn.init.trunc_normal_(self.global_token, std=0.02)

        if self.loss_type == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.loss_type == 'FOCAL':
            self.criterion = FocalLoss()
        elif self.loss_type == "ASYM":
            self.criterion = AsymmetricLoss()
        elif self.loss_type == 'FOCAL_SVDD':
            self.criterion = FocalLoss()
            self.svdd_criterion = MultiCenterSVDDLoss(3, cfg.num_heads)
        elif self.loss_type == "ASYM_SVDD":
            self.criterion = AsymmetricLoss()
            self.svdd_criterion = MultiCenterSVDDLoss(3, cfg.num_heads)
        elif self.loss_type == "FOCAL_CENTER":
            self.criterion = FocalLoss()
            self.svdd_criterion = CenterLoss(3, 64)
        elif self.loss_type == "ASYM_CENTER":
            self.criterion = AsymmetricLoss()
            self.svdd_criterion = CenterLoss(3, 64)
        
        self._log_cosine = getattr(cfg, "log_cosine_per_layer", False)
        self._cos_logs = []  # list of floats per layer for last forward
        self._cos_sample_cap = int(getattr(cfg, "cos_sample_cap", 4096))  # subsample nodes for speed
    # ------------------------------------------------------------------
    def forward(self, data):
        """Expected fields on `data` (PyG Batch):
        x           – node features       [N, node_in_dim]
        edge_index  – COO index           [2, E]
        edge_attr   – edge features       [E, edge_in_dim]
        attn_bias   – attention bias      [N, N]  (optional, broadcastable)
        batch       – graph assignment    [N]
        """
        data.x = data.x.long()
        data.edge_attr = data.edge_attr.long()
        data.face_attr = data.face_attr.long()
        data.global_idx = data.global_idx.long()

        self._cos_logs = []
        x_chem_embed = self.node_chem_embedder(data.x)
        x_chem_feats = self.node_chem_encoder(x_chem_embed)
        x_local_encoding = self.xlocal_encoder(data.edge_index, data.num_nodes)
        if self.cfg.barycentric:
            x_vec = self.node_laplace_vec(data.bary_nodevec)
            edge_vec = self.edge_laplace_vec(data.bary_edgevec)
            ring_vec = self.ring_laplace_vec(data.bary_ringvec)
            x_rw_encoding = self.node_rw_encoder(data.bary_noderw)
            edge_rw_encoding = self.edge_rw_encoder(data.bary_edgerw)
            ring_rw_encoding = self.ring_rw_encoder(data.bary_ringrw)
        else:
            x_val_batch = self.laplace_val(data.eigenval)
            x_val = x_val_batch[ data.batch ]
            x_vec = self.laplace_vec(data.eigenvec)
            x_rw_encoding = self.random_walk_encoder(data.random_walk)
        if self.position:
            kernel_vector = self.distance_embedder(data.edge_index, data.pos)  # [E, K]
            x3d_encoding = self.x3d_encoder(kernel_vector, data.edge_index, data.num_nodes)
            if self.cfg.barycentric:
                x = torch.cat([x_chem_feats, x_vec, x_rw_encoding, x_local_encoding, x3d_encoding], dim=-1)
            else:
                x = torch.cat([x_chem_feats, x_vec, x_val, x_rw_encoding, x_local_encoding, x3d_encoding], dim=-1)
        else:
            if self.cfg.barycentric:
                x = torch.cat([x_chem_feats, x_vec, x_rw_encoding, x_local_encoding], dim=-1)    
            else:
                x = torch.cat([x_chem_feats, x_vec, x_val, x_rw_encoding, x_local_encoding], dim=-1)           
        x = self.project_x(x)

        edge_chem_embed = self.edge_chem_embedder(data.edge_attr)
        edge_chem_feats = self.edge_chem_encoder(edge_chem_embed)
        if self.position:
            bl_encoding = self.bl_encoder(kernel_vector)  # [E, 32]
            edge_attr = torch.cat([edge_chem_feats, bl_encoding], dim=-1)
        else:
            if self.cfg.barycentric:
                edge_attr = torch.cat([edge_chem_feats, edge_vec, edge_rw_encoding], dim=-1)
            else:
                edge_attr = torch.cat([edge_chem_feats], dim=-1)
        edge_attr = self.project_edge(edge_attr)

        if self.cfg.face:
            face_chem_embed = self.face_chem_embedder(data.face_attr)
            face_chem_feats = self.face_chem_encoder(face_chem_embed)
            if self.cfg.barycentric:
                face_attr = torch.cat([face_chem_feats, ring_vec, ring_rw_encoding], dim=-1)
                face_attr = self.project_face(face_attr)
            else:
                face_attr = torch.cat([face_chem_feats], dim=-1)
            face_attr = self._maybe_add_face_noise(face_attr)
        else:
            face_attr = data.face_attr
        
        if self.cfg.barycentric:
            global_embed = self.global_laplace_val(data.bary_eigvals)
        else:
            global_embed = self.global_encoder(data.global_idx)

        if self.attention:
            # 4) Replace the list with your batched tensor
            if self.cfg.barycentric:
                spd_bias_tensor = self.spd_embedder(data.bary_spd)
            else:
                spd_bias_tensor = self.spd_embedder(data.spd)
            
            attn_biased = spd_bias_tensor if self.cfg.spd_bias else None
            for layer in self.layers:
                x, edge_attr, face_attr, global_embed = layer(
                    x, data.edge_index, edge_attr, data.face_index, face_attr, global_embed, data.batch, attn_biased
                )
                if self._log_cosine:
                    with torch.no_grad():
                        val = self._mean_pairwise_cosine(x, self._cos_sample_cap)
                        self._cos_logs.append(val.item())
        
        else:
            for layer in self.layers:
                x, edge_attr, face_attr, global_embed = layer(
                    x, data.edge_index, edge_attr, data.face_index, face_attr, global_embed, data.batch
                )
                if self._log_cosine:
                    with torch.no_grad():
                        val = self._mean_pairwise_cosine(x, self._cos_sample_cap)
                        self._cos_logs.append(val.item())

        # x = self.pool(x, data.batch, target_embed=getattr(data, 'protein_embed', None))
        x = global_mean_pool(x, data.batch)
        logits = self.decoder(x)
        if self.loss_type == "FOCAL_SVDD" or self.loss_type == "ASYM_SVDD" or self.loss_type == "FOCAL_CENTER" or self.loss_type == "ASYM_CENTER":
            svdd_embedding = self.svdd_decoder(x)
        output = {}
        if 'loss' in self.output_type:
            target = data.y
            with autocast(device_type=cf.DEVICE, enabled=False):
                regular_loss = self.criterion(logits.float(), target.float())
                if self.loss_type == "FOCAL_SVDD" or self.loss_type == 'ASYM_SVDD' or self.loss_type == "FOCAL_CENTER" or self.loss_type == "ASYM_CENTER":
                    svdd_loss = self.svdd_criterion(svdd_embedding, target.float())
                    output["svdd"] = svdd_loss
                    output['losses'] = regular_loss + self.cfg.lambda_ * svdd_loss
                else:
                    output['losses'] = regular_loss
        if 'infer' in self.output_type:
            output['bind'] = torch.sigmoid(logits)
        return output

    def enable_attn_logging(self, flag: bool = True):
        """Turn per-layer attention logging on/off."""
        if not getattr(self, "attention", False):
            return
        self._log_attention = flag
        for block in self.layers:
            if hasattr(block, "attn"):
                block.attn._log_attn = flag

    def clear_attn_logs(self):
        """Clear accumulated attention stats in the logger."""
        if not hasattr(self, "_attn_logger"):
            return
        # If your logger has reset(), prefer it
        if hasattr(self._attn_logger, "reset"):
            self._attn_logger.reset()
            return
        # Fallback: manually clear buffers
        bufs = getattr(self._attn_logger, "buffers", None)
        if bufs is not None:
            for buf in bufs:
                buf.ent.clear()
                buf.long_ratio.clear()
                buf.pearson.clear()

    def get_attn_log_summary(self):
        """Return a dict of averaged per-layer stats from the logger."""
        if not getattr(self, "attention", False) or not hasattr(self, "_attn_logger"):
            return {}
        return self._attn_logger.summarize()
    
    @torch.no_grad()
    def _maybe_add_face_noise(self, face_attr: torch.Tensor) -> torch.Tensor:
        # only at eval, and only if sigma>0 and faces are enabled
        if not self.cfg.face:
            return face_attr
        if self.training and self.enable_face_noise_only_eval:
            return face_attr
        sigma = float(self.eval_face_noise_sigma)
        if sigma <= 0.0:
            return face_attr
        # feature-wise std (avoid div by 0)
        std = face_attr.std(dim=0, keepdim=True).clamp_min(1e-8)
        noise = torch.randn_like(face_attr) * (sigma * std)
        return face_attr + noise
    
    def set_face_alpha(self, alpha: float):
        for blk in self.layers:
            blk.mpn.set_face_alpha(alpha)
    
    @staticmethod
    def _mean_pairwise_cosine(x: torch.Tensor, sample_cap: int = 4096) -> torch.Tensor:
        # x: [N, D], returns scalar tensor
        N = x.size(0)
        if N <= 1:
            return x.new_tensor(1.0)
        if N > sample_cap:
            idx = torch.randperm(N, device=x.device)[:sample_cap]
            x = x.index_select(0, idx)
            N = x.size(0)
        x = F.normalize(x, dim=-1)
        # compute upper-triangular mean efficiently
        sim = x @ x.t()
        iu = torch.triu_indices(N, N, offset=1, device=x.device)
        return sim[iu[0], iu[1]].mean()

    def enable_cosine_logging(self, flag: bool = True):
        self._log_cosine = bool(flag)

    def get_cosine_logs(self) -> list[float]:
        # returns last forward’s per-layer means
        return [float(v) for v in self._cos_logs]
    
    def enable_attn_debug_print(self, flag: bool = True, layers: list[int] | None = None):
        """
        Toggle simple print-based attention debugging inside each attention layer.
        If `layers` is provided, only those layer indices are toggled.
        """
        if not getattr(self, "attention", False):
            return
        self._attn_debug_print = bool(flag)
        idxs = range(self.num_layers) if layers is None else layers
        for li in idxs:
            if li < 0 or li >= self.num_layers:
                continue
            block = self.layers[li]
            if hasattr(block, "attn"):
                # create attr if missing; safe for any attn impl
                setattr(block.attn, "debug_print_attn", bool(flag))
                # ensure layer index is present for nicer logs
                if getattr(block.attn, "_layer_idx", None) is None:
                    block.attn._layer_idx = li

    def disable_attn_debug_print(self):
        """Shorthand to turn off print debugging."""
        self.enable_attn_debug_print(False)