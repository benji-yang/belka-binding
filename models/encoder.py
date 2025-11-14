import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import degree as tgdegree
from torch_geometric.utils import to_dense_adj, to_dense_batch


def _maybe_contiguous(x):
    return x if x.is_contiguous() else x.contiguous()


class Global3DCentralityEncodingSparse(nn.Module):
    """
    Efficient: avoids an intermediate allocation by passing `out=` into scatter_add,
    and avoids a wasted zeros(...) that was never used.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self,
                psi: torch.Tensor,            # (E, K), float
                edge_index: torch.LongTensor, # (2, E), long
                num_nodes: int) -> torch.Tensor:
        row = edge_index[0]  # (E,)
        K   = psi.shape[1]

        # Preallocate output once; fill with scatter (no temp)
        sum_psi = psi.new_zeros((num_nodes, K))
        scatter_add(src=_maybe_contiguous(psi),
                    index=_maybe_contiguous(row),
                    dim=0,
                    out=sum_psi)

        # Linear prefers contiguous memory
        x3d = self.linear(_maybe_contiguous(sum_psi))
        return x3d


class GlobalEncoder(nn.Module):
    def __init__(self, num_emb : int = 1, global_dim: int = 64):
        super().__init__() 
        self.embed = nn.Embedding(num_emb, global_dim)

    
    def forward(self, global_state):
        return self.embed(global_state)


class CentEncoder(nn.Module):
    def __init__(self, max_degree: int, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(max_degree + 1, embed_dim)

    def forward(self, edge_index: torch.LongTensor, num_nodes: int):
        # Compute degree on CPU/GPU depending on input; request long directly
        deg = tgdegree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        deg.clamp_(max=self.embed.num_embeddings - 1)  # in-place clamp
        return self.embed(deg)


class NodeEncoder(nn.Module):
    def __init__(self, n_atom_types: int, n_period: int, n_group: int, n_degree: int, n_valence: int, 
                 n_h_counts: int, n_radical_charges: int, n_formal_charges: int, n_hybs: int, 
                 embed_dim: int = 64, boolean_dim: int = 2):
        super().__init__()
        self.atom_type_embed = nn.Embedding(n_atom_types, embed_dim)
        self.period_embed = nn.Embedding(n_period, embed_dim)
        self.group_embed = nn.Embedding(n_group, embed_dim)
        self.degree_embed = nn.Embedding(n_degree, embed_dim)
        self.valence_embed = nn.Embedding(n_valence, embed_dim)
        self.h_embed = nn.Embedding(n_h_counts, embed_dim)
        self.radical_embed = nn.Embedding(n_radical_charges, embed_dim)
        self.charge_embed = nn.Embedding(n_formal_charges, embed_dim)
        self.hybs_embed = nn.Embedding(n_hybs, embed_dim)
        self.aromatic_embed = nn.Embedding(boolean_dim, embed_dim)
        self.ring_embed = nn.Embedding(boolean_dim, embed_dim)
        self.chiral_embed = nn.Embedding(boolean_dim, embed_dim)

    def forward(self, x_tensor: torch.LongTensor):
        # Size [N_node, 64]
        node_vector = (self.atom_type_embed(x_tensor[:,0]) + self.period_embed(x_tensor[:,1]) + 
                        self.group_embed(x_tensor[:,2]) + self.degree_embed(x_tensor[:,3]) + 
                        self.valence_embed(x_tensor[:,4]) + self.h_embed(x_tensor[:,5]) + 
                        self.radical_embed(x_tensor[:,6]) + self.charge_embed(x_tensor[:,7]) +
                        self.hybs_embed(x_tensor[:, 8]) + self.aromatic_embed(x_tensor[:, 9]) +
                        self.ring_embed(x_tensor[:, 10]) + self.chiral_embed(x_tensor[:, 11]))
        return node_vector


class EdgeEncoder(nn.Module):
    def __init__(self, n_bond_type: int, n_stereo_type: int, embed_dim: int = 64, boolean_dim: int = 2):
        super().__init__()
        self.bond_embed = nn.Embedding(n_bond_type, embed_dim)
        self.stereo_embed = nn.Embedding(n_stereo_type, embed_dim)
        self.conjugated_embed = nn.Embedding(boolean_dim, embed_dim)
        self.in_ring_embed = nn.Embedding(boolean_dim, embed_dim)

    def forward(self, edge_tensor: torch.LongTensor):
        edge_vector = (self.bond_embed(edge_tensor[:, 0]) + self.stereo_embed(edge_tensor[:, 1]) +
                       self.conjugated_embed(edge_tensor[:, 2]) + self.in_ring_embed(edge_tensor[:, 3]))
        return edge_vector


class RingEncoder(nn.Module):
    def __init__(self, n_ring_size: int, n_het_counts: int, n_avg_en: int, embed_dim: int = 64, boolean_dim: int = 2):
        super().__init__()
        self.ring_size_embed = nn.Embedding(n_ring_size, embed_dim)
        self.aromatic_embed = nn.Embedding(boolean_dim, embed_dim)
        self.het_counts_embed = nn.Embedding(n_het_counts, embed_dim)
        self.saturated_embed = nn.Embedding(boolean_dim, embed_dim)
        self.fusion_embed = nn.Embedding(boolean_dim, embed_dim)
        self.avg_en_embed = nn.Embedding(n_avg_en, embed_dim)

    def forward(self, face_tensor: torch.LongTensor):
        face_vector = (self.ring_size_embed(face_tensor[:, 0]) + self.aromatic_embed(face_tensor[:, 1]) +
                       self.het_counts_embed(face_tensor[:, 2]) + self.saturated_embed(face_tensor[:, 3]) +
                       self.fusion_embed(face_tensor[:, 4]) + self.avg_en_embed(face_tensor[:, 5]))
        return face_vector


class LaplacianPEVec(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        self.vec_mlp = MLPEncoder(k)

    def forward(self, x):
        """
        x: [..., k]
        For training, apply random sign flips (in-place for speed if allowed).
        """
        if self.training:
            # Create ±1 with Bernoulli on a small view, then broadcast multiply in-place
            # Matches your original semantics of per-eigenvector sign per sample.
            sign = torch.empty_like(x).bernoulli_().mul_(2).sub_(1)  # {0,1} -> { -1, +1 }
            x = x.mul(sign)  # avoids a new allocation vs x * sign
        return self.vec_mlp(x)


class LaplacianPEVal(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        self.val_mlp = MLPEncoder(k)

    def forward(self, x):
        return self.val_mlp(x)


class GlobalLaplacianPEVal(nn.Module):
    def __init__(self, k=7, global_dim=64):
        super().__init__()
        self.k = k
        self.val_mlp = MLPEncoder(k, out_dim=global_dim)

    def forward(self, x):
        return self.val_mlp(x)


class RandomWalkStructuralEncoding(nn.Module):
    def __init__(self, k_rw: int = 16):
        super().__init__()
        self.k_rw = k_rw
        self.mlp  = MLPEncoder(k_rw)

    def forward(self, x):
        return self.mlp(x)


class SPDAttentionBias(nn.Module):
    def __init__(self, heads: int = 32, max_distance: int = 50):
        """
        heads:         number of attention heads
        max_distance: clip all distances > max_distance to max_distance
                       (this also defines the size of the embedding table)
        """
        super().__init__()
        self.heads = heads
        self.overflow_bucket = max_distance + 1
        self.pad_bucket = max_distance + 2
        self.hop_bias = nn.Embedding(max_distance + 3, self.heads, padding_idx=self.pad_bucket)
        # NEW: ephemeral cache (not parameters/buffers)
        self._last_idx = None   # [B,N,N] Long
        self._last_mask = None  # [B,N] Bool (valid nodes)

    def forward(self, matrix):
        # matrix size [B, Nmax, Nmax]
        # 5) clamp distances to [0..max_distance]
        idx = torch.where(
            matrix < 0,
            torch.full_like(matrix, self.pad_bucket),
            matrix.clamp_max(self.overflow_bucket)
        ).long()  # [B,N,N]

        # 6) turn into biases: embedding returns [B, Nmax, Nmax, heads]
        B_spd = self.hop_bias(idx)  # Long→Float
        mask3d = (matrix == -1)
        mask4d = mask3d.unsqueeze(-1)  
        B_spd[mask4d.expand_as(B_spd)] = 0   

        # 7) permute to [B, heads, Nmax, Nmax] to match multi‐head bias
        B_spd = B_spd.permute(0, 3, 1, 2)

        valid_nodes = ~(mask3d.any(dim=-1))  # [B,N] True for real nodes
        self._last_idx = idx.detach()
        self._last_mask = valid_nodes.detach()

        return B_spd

    # Helper to get dense SPD with pads as -1 (match your input convention)
    def get_last_spd_dense(self):
        if self._last_idx is None:
            return None, None
        spd = self._last_idx.clone()
        spd[spd == self.pad_bucket] = -1  # restore pads
        spd[spd == self.overflow_bucket] = self.overflow_bucket  # keep overflow as capped
        return spd, self._last_mask

def get_attn_masks(shortest_path_feats: torch.Tensor, activation_dtype: torch.dtype):
    """
    Generate attention masks and biases based on shortest-path features.

    Args:
      shortest_path_feats: Tensor of shape [N, N] or [B, N, N] with values:
        -1 indicates no connection (mask out)
        >=0 indicates distance offset by 1 (0 means original distance 1)
      activation_dtype: torch.float32 or torch.float16

    Returns:
      attn_mask: Tensor same shape as feats[..., None] with 1 for valid attention, 0 for masked
      attn_bias: Tensor same shape as feats[..., None] with 0 for valid, -inf (or -10000) for masked
    """
    # Determine minus infinity value
    if activation_dtype == torch.float32:
        minus_inf_val = float('-inf')
    else:
        minus_inf_val = -10000.0
    minus_inf = torch.tensor(minus_inf_val, dtype=activation_dtype, device=shortest_path_feats.device)

    # Build mask and bias
    # feats[..., None] aligns dims: [..., 1]
    valid = shortest_path_feats[..., None] != -1
    attn_mask = valid.to(activation_dtype)
    attn_bias = torch.where(valid, torch.tensor(0.0, dtype=activation_dtype, device=shortest_path_feats.device), minus_inf)
    return attn_mask, attn_bias


class MLPEncoder(nn.Module):
    """
    Same topology as before but uses functional LN for fewer modules, keeps things contiguous,
    and uses in-place where safe. Still Dropout for regularization.
    """
    def __init__(self, in_dim: int, hidden_mul: int = 2, out_dim: int = 32, dropout: float = 0.18):
        super().__init__()
        self.in_dim     = int(in_dim)
        self.hidden_dim = int(hidden_mul * in_dim)
        self.out_dim    = int(out_dim)
        self.dropout_p  = float(dropout)

        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

        # Pre-register eps for layer_norm
        self.ln_eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm → Linear → ReLU
        x = F.layer_norm(x, normalized_shape=(self.in_dim,), eps=self.ln_eps)
        x = self.fc1(x)
        x = F.relu(x)  # in-place

        # LayerNorm → Linear → Dropout
        x = F.layer_norm(x, normalized_shape=(self.hidden_dim,), eps=self.ln_eps)
        x = self.fc2(x)
        if self.training and self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        return x


# class SPDAttentionBias(nn.Module):
#     def __init__(self, heads: int = 32, max_distance: int = 80):
#         """
#         heads:         number of attention heads
#         max_distance: clip all distances > max_distance to max_distance
#                        (this also defines the size of the embedding table)
#         """
#         super().__init__()
#         self.heads = heads
#         self.max_distance = max_distance
#         # distances 0,1,2,...,max_distance
#         self.embed = nn.Embedding(max_distance + 2, heads, padding_idx=0)
#         # num embeddings is max+2 to account for clipped values

#     def forward(self, adj, mask, batch):
#         # data.edge_index: LongTensor [2, E]
#         # data.batch:      LongTensor [N_tot] with values in [0..B-1]
#         B      = int(batch.max().item()) + 1

#         # 1) build a batched dense adjacency [B, Nmax, Nmax]
#         Nmax = adj.size(1)

#         # 3) init distance tensor D[b,i,j]:=∞ except
#         #    D[b,i,i]=0, D[b,i,j]=1 if edge exists
#         INF = -2  # any path length ≤ Nmax, will become -1 after D += 1 at below
#         D = torch.full((B, Nmax, Nmax), INF, device=adj.device, dtype=torch.long)

#         # zero on the diagonal
#         eye = torch.arange(Nmax)
#         D[:, eye, eye] = 0

#         # one for each real edge
#         # A>0 is [B,Nmax,Nmax] bool
#         D[adj > 0] = 1

#         # 4) batched Floyd–Warshall
#         #    D[b,i,j] = min(D[b,i,j], D[b,i,k]+D[b,k,j])
#         for k in range(Nmax):
#             # shapes: D[:,:,k] → [B, Nmax], D[:,k,:] → [B, Nmax]
#             # unsqueeze to [B, Nmax, 1] and [B, 1, Nmax] to broadcast
#             alt = D[:, :, k].unsqueeze(2) + D[:, k, :].unsqueeze(1)
#             D = torch.min(D, alt)

#         D += 1
#         real = mask.view(B, Nmax, 1) & mask.view(B, 1, Nmax)
#         D[~real] = 0

#         # 5) clamp distances to [0..max_distance]
#         distances = D.clamp(min=0, max=self.max_distance+1)

#         # 6) turn into biases: embedding returns [B, Nmax, Nmax, heads]
#         B_spd = self.embed(distances)  # Long→Float
#         mask3d = (D == -1)
#         mask4d = mask3d.unsqueeze(-1)  
#         B_spd[mask4d.expand_as(B_spd)] = 0   

#         # 7) permute to [B, heads, Nmax, Nmax] to match multi‐head bias
#         B_spd = B_spd.permute(0, 3, 1, 2)

#         # (optional) zero out padded rows/cols so they don’t interfere:
#         # row_mask = mask.unsqueeze(1).unsqueeze(-1)
#         # col_mask = mask.unsqueeze(1).unsqueeze(2)
#         # B_spd = B_spd * (row_mask & col_mask).to(B_spd.dtype)

#         return B_spd

# class RandomWalkStructuralEncoding(nn.Module):
#     def __init__(self, k_rw: int = 16):
#         """
#         k_rw:       number of steps in the random walk (Eq. 30 uses 16)
#         hidden_dim: output dimension of the MLP encoder (32)
#         """
#         super().__init__()
#         self.k_rw = k_rw
#         # the MLP that maps the k_rw–dim RW features into hidden_dim
#         self.mlp = MLPEncoder(k_rw)

#     def forward(self, adj, mask, batch):
#         """
#         data: a torch_geometric.data.Batch with
#             - data.edge_index: LongTensor [2, E]
#             - data.batch:      LongTensor [N_tot] mapping each node to its graph in [0..B-1]
#         returns:
#             x_rw:  FloatTensor [N_tot, hidden_dim]
#         """
#         B = int(batch.max().item()) + 1

#         # 2) compute per‐node degree and normalized transition P = D^{-1} A
#         deg = adj.sum(dim=-1)                                         # [B, Nmax]
#         # avoid division by zero: isolated nodes get zero rows in P
#         deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
#         P = deg_inv.unsqueeze(-1) * adj                               # [B, Nmax, Nmax]

#         # 3) iteratively compute P^k and collect diag entries
#         Pk = P.clone()                                              # P^1
#         diags = [Pk.diagonal(dim1=1, dim2=2)]                       # list of [B, Nmax]
#         for _ in range(1, self.k_rw):
#             Pk = Pk.bmm(P)                                          # P^{k+1}
#             diags.append(Pk.diagonal(dim1=1, dim2=2))

#         # stack → [B, Nmax, k_rw]
#         rw_feats = torch.stack(diags, dim=2)

#         # 4) build mask of “real” nodes vs padding

#         # 5) gather only the real nodes into a flat tensor [N_tot, k_rw]
#         flat_feats = rw_feats[mask]                                 # (sum_n N_graph, k_rw)

#         # 6) apply the MLP to get [N_tot, hidden_dim]
#         x_rw = self.mlp(flat_feats)

#         return x_rw

# class LaplacianPositionalEncoding(nn.Module):
#     def __init__(self, k=7):
#         """
#         k: number of non-trivial eigenpairs to keep (the paper uses k=7)
#         hidden_dim: output dimensionality of each MLP
#         """
#         super().__init__()
#         self.k = k
#         # MLP for eigenvectors (Eq. 28)
#         self.vec_mlp = MLPEncoder(k)
#         # MLP for eigenvalues (Eq. 29)
#         self.val_mlp = MLPEncoder(k)
#         self.norm = 'L2'
#         self.eps = 1e-12

#     def forward(self, adj, mask, batch):
#         # data.edge_index: (2, E)
#         # data.batch:     (N_tot,) with values in [0..B-1]
#         B = int(batch.max()) + 1

#         # 2) build a [B, Nmax] mask of real vs. padded nodes

#         # 3) unnormalized Laplacian L = D - A
#         deg = adj.sum(dim=-1)                                              # [B, Nmax]
#         D   = torch.diag_embed(deg)                                      # [B, Nmax, Nmax]
#         L   = D - adj                                                      # [B, Nmax, Nmax]

#         # 4) add large diagonal on padded rows so their eigenvalues sit at the top
#         #    choose M > any λ_max(L) ⪅ 2 * max_degree
#         max_deg = deg.max(dim=1).values                                 # [B]
#         M       = max_deg * 2 + 1                                        # [B]
#         pad_mask = (~mask).float()                                       # [B, Nmax]
#         L = L + torch.diag_embed(pad_mask * M.unsqueeze(1))              # [B, Nmax, Nmax]

#         # 5) batched eigendecomposition
#         #    lambdas: [B, Nmax],  U: [B, Nmax, Nmax]
#         lambdas, U = torch.linalg.eigh(L)

#         # 6) slice out the 1…k smallest non‐trivial eigenpairs
#         U_k      = U[:, :, 1 : 1 + self.k]                               # [B, Nmax, k]
#         lambdas_k = lambdas[:, 1 : 1 + self.k]                           # [B, k]

#         # 5) Normalize each eigenvector “column” per graph
#         if self.norm == "L2":
#             denom = torch.linalg.norm(U_k, dim=1)   # [B, k]
#         elif self.norm == "L1":
#             denom = U_k.abs().sum(dim=1)            # [B, k]
#         elif self.norm == "abs-max":
#             denom = U_k.abs().amax(dim=1).values    # [B, k]
#         else:
#             raise ValueError(f"Unsupported normalization `{self.norm}`")

#         denom = denom.clamp(min=self.eps)           # avoid divide‑by‑zero
#         U_k = U_k / denom.unsqueeze(1)              # [B, Nmax, k]

#         # 7) random sign‐flip per graph (only in training)
#         if self.training:
#             sign = (torch.randint(0, 2, (B, self.k), device=adj.device, dtype=torch.float32) * 2 - 1)
#             # sign[b,i] is ±1
#             U_k = U_k * sign.unsqueeze(1)                                # broadcast over the node dimension

#         # 8) apply the two MLPs in batch
#         #    vec_mlp: input [..., k] → output [..., hidden_dim]
#         x_vec = self.vec_mlp(U_k)                                        # [B, Nmax, hidden_dim]

#         #    normalize each graph’s eigenvalues, then MLP → [B, hidden_dim]
#         λ_norm = lambdas_k / (lambdas_k.norm(dim=1, keepdim=True) + 1e-12)
#         x_val_graph = self.val_mlp(λ_norm)                               # [B, hidden_dim]

#         #    broadcast each graph’s x_val across its nodes
#         x_val = x_val_graph.unsqueeze(1).expand(-1, U_k.size(1), -1)     # [B, Nmax, hidden_dim]

#         # # 9) combine
#         # x_pe_batch = x_vec + x_val                                        # [B, Nmax, hidden_dim]

#         # 10) scatter back to a flat (N_tot, hidden_dim) tensor
#         out_vec, out_val = [], []
#         for b in range(B):
#             mb = mask[b]                                                  # (Nmax,) bool
#             # out_pe .append(x_pe_batch[b, mb])
#             out_vec.append(x_vec[b,    mb])
#             out_val.append(x_val[b,    mb])
#         # x_pe  = torch.cat(out_pe,  dim=0)                                 # (N_tot, hidden_dim)
#         x_vec = torch.cat(out_vec, dim=0)
#         x_val = torch.cat(out_val, dim=0)

#         return x_vec, x_val