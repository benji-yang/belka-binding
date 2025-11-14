from __future__ import annotations
import torch
from dataclasses import dataclass, field

def align_spd_to_attn(spd_batch: torch.Tensor, qmask: torch.Tensor, fill: int = -1) -> torch.Tensor:
    # spd_batch: [B, Ns, Ns] (−1 for pad/unreachable), qmask: [B, Na]
    B, Ns, _ = spd_batch.shape
    Na = qmask.size(1)
    if Ns == Na:
        spd = spd_batch
    elif Ns > Na:
        spd = spd_batch[:, :Na, :Na]
    else:
        pad = (0, Na - Ns, 0, Na - Ns)  # (right, left, bottom, top) for last two dims
        spd = torch.nn.functional.pad(spd_batch, pad, value=fill)

    # apply qmask (invalidate rows/cols for padded queries)
    inv = ~qmask.bool()
    spd = spd.clone()
    spd[inv.unsqueeze(-1).expand(-1, -1, Na)] = fill
    spd[inv.unsqueeze(-2).expand(-1, Na, -1)] = fill
    return spd

def _attn_entropy(
    A: torch.Tensor,
    qmask: torch.Tensor | None,
    valid_keys: torch.Tensor | None = None,   # NEW
) -> torch.Tensor:
    """
    A: [B,H,N,N] post-softmax (padded rows/cols zeroed)
    qmask: [B,N] valid queries (from to_dense_batch)
    valid_keys: [B,N,N] bool mask of valid pairs (query i, key j).  # NEW
                 If None, inferred from A>0 (fallback).
    returns: [H] mean normalized entropy per head
    """
    eps = 1e-12
    A = A.clamp_min(eps)
    B, H, N, _ = A.shape
    device = A.device

    if valid_keys is None:
        # Fallback: anything with positive mass is considered valid
        valid_keys = (A.sum(dim=1) > 0)  # [B,N,N]
    # Exclude diagonal (self keys) so entropy matches ratios
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1,N,N]
    valid_keys = valid_keys & (~eye)

    # Per-row entropy
    H_row = -(A * A.log()).sum(dim=-1)                       # [B,H,N]

    # Count valid keys per (b,i) row (how many j are valid for query i)
    K_row = valid_keys.to(A.dtype).sum(dim=-1)                             # [B,N]
    mask_gt1 = (K_row > 1)

    # denom = log(K_row) for rows with >1 key, else 1.0 (so those rows contribute 0)
    denom = torch.ones_like(K_row)
    denom[mask_gt1] = K_row[mask_gt1].log()

    # broadcast to [B,H,N]
    H_norm = H_row / denom.unsqueeze(1)
    # zero rows with <=1 valid key
    H_norm = H_norm.masked_fill(~mask_gt1.unsqueeze(1), 0.0)

    if qmask is not None:
        qm = qmask.unsqueeze(1)               # [B,1,N]
        num = (H_norm * qm).sum(dim=(0,2))    # [H]
        den = qm.sum(dim=(0,2)).clamp_min(1)  # scalar -> broadcast OK
        return num / den
    return H_norm.mean(dim=(0,2))

def _attn_distance_profile(
    A: torch.Tensor,
    spd: torch.Tensor,
    qmask: torch.Tensor | None,
    overflow_bucket: int,
    valid_keys: torch.Tensor | None = None,   # NEW (optional)
):
    """
    Returns same outputs as before.
    If valid_keys is given, uses it; otherwise computes it internally.
    """
    B, H, N, _ = A.shape
    device = A.device

    # SPD preprocessing
    spd_eff = torch.where(spd == -1, spd, spd.clamp(min=0, max=overflow_bucket))

    # Build valid mask if not provided: reachable + (optional) q/k validity
    if valid_keys is None:
        valid = (spd_eff != -1)  # reachable
        if qmask is not None:
            qv = qmask.unsqueeze(-1)  # [B,N,1]
            kv = qmask.unsqueeze(-2)  # [B,1,N]
            valid = valid & qv & kv
    else:
        valid = valid_keys

    dmax = int(torch.clamp(spd_eff, min=0).max().item())
    dmax = max(dmax, 0)

    non_diag = (~torch.eye(N, dtype=torch.bool, device=device)).unsqueeze(0)  # [1,N,N]

    mean_by_d = []
    ratio_long_8  = []
    ratio_long_16 = []
    pearson_r = []

    # Flatten helpers
    valid_flat = valid.view(B, -1)                         # [B, N*N]
    spd_flat   = spd_eff.view(B, -1)                       # [B, N*N]
    offdiag    = non_diag.expand(B, -1, -1).reshape(B, -1) # [B, N*N]

    # Masks for pair-weighted ratios (non-self pairs)
    sel_pos  = valid_flat & (spd_flat > 0)
    sel_d8   = valid_flat & (spd_flat >= 8)
    sel_d16  = valid_flat & (spd_flat >= 16)

    for h in range(H):
        Ah = A[:, h]                  # [B,N,N]
        Ah_flat = Ah.reshape(B, -1)

        # Mean attention by exact distance
        sums = torch.zeros(dmax + 1, device=device, dtype=A.dtype)
        cnts = torch.zeros(dmax + 1, device=device, dtype=A.dtype)
        for d in range(dmax + 1):
            sel = valid & (spd_eff == d)        # [B,N,N]
            if sel.any():
                sums[d] = Ah[sel].sum()
                cnts[d] = sel.sum().to(A.dtype)
        mean_d = torch.where(cnts > 0, sums / cnts, torch.zeros_like(sums))
        mean_by_d.append(mean_d)

        # Pair-weighted long-range mass ratios
        den  = Ah_flat[sel_pos].sum().clamp_min(1e-12)
        num8 = Ah_flat[sel_d8].sum()
        num16= Ah_flat[sel_d16].sum()
        ratio_long_8.append(num8  / den)
        ratio_long_16.append(num16 / den)

        # Pearson over valid off-diagonals
        sel_od = (valid_flat & offdiag)
        if sel_od.any():
            a_vec = Ah_flat[sel_od].to(torch.float32)
            d_vec = spd_flat[sel_od].to(torch.float32)
            a_c = a_vec - a_vec.mean()
            d_c = d_vec - d_vec.mean()
            denom_pd = (a_c.norm() * d_c.norm()).clamp_min(1e-12)
            r = (a_c * d_c).sum() / denom_pd
        else:
            r = torch.tensor(0.0, device=device)
        pearson_r.append(r)

    return (
        mean_by_d,
        torch.stack(ratio_long_8,  dim=0),
        torch.stack(ratio_long_16, dim=0),
        torch.stack(pearson_r,     dim=0),
    )


@dataclass
class _LayerBuf:
    ent: list = field(default_factory=list)           # list of [H] tensors
    long_ratio_8: list = field(default_factory=list)  # list of [H] tensors
    long_ratio_16: list = field(default_factory=list) # list of [H] tensors
    pearson: list = field(default_factory=list)       # list of [H] tensors
    # you can add more if needed


class MultiLayerAttnLogger:
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.buffers = [_LayerBuf() for _ in range(n_layers)]
        self._spd_overflow_bucket = None

    def set_overflow_bucket(self, overflow_bucket: int):
        self._spd_overflow_bucket = overflow_bucket

    @torch.no_grad()
    def update(self, layer_idx: int, A: torch.Tensor, qmask: torch.Tensor, spd_dense: torch.Tensor):
        """
        A:        [B,H,N,N] post-softmax attention (padded queries/keys set to 0)
        qmask:    [B,N] valid nodes mask from to_dense_batch
        spd_dense:[B,N,N] SPD with -1 for unreachable/pad, same B/N as A
        """
        # Build one consistent valid_keys mask
        B, H, N, _ = A.shape
        spd_aligned = align_spd_to_attn(spd_dense, qmask, fill=-1)   # -> [B,N,N]
        overflow = self._spd_overflow_bucket or 999
        spd_eff = torch.where(spd_aligned == -1, spd_aligned, spd_aligned.clamp(min=0, max=overflow))

        valid_keys = (spd_eff != -1) & qmask.unsqueeze(-1) & qmask.unsqueeze(-2)
        eye = torch.eye(N, dtype=torch.bool, device=A.device).unsqueeze(0)
        valid_keys = valid_keys & (~eye)

        # sanity
        assert valid_keys.shape == (B, N, N), (valid_keys.shape, (B, N, N))
        # Entropy per head (use the SAME valid_keys for normalization)
        ent_h = _attn_entropy(A, qmask, valid_keys=valid_keys)  # [H]

        # Distance profile per head (new: returns ratio@8 and ratio@16)
        mean_by_d, ratio8_h, ratio16_h, pearson_h = _attn_distance_profile(
            A, spd_dense, qmask, self._spd_overflow_bucket or 999, valid_keys=valid_keys
        )

        buf = self.buffers[layer_idx]
        buf.ent.append(ent_h.cpu())
        buf.long_ratio_8.append(ratio8_h.cpu())
        buf.long_ratio_16.append(ratio16_h.cpu())
        buf.pearson.append(pearson_h.cpu())

    def summarize(self):
        """
        Returns dict of lists (length = n_layers) of [H]-tensors (averaged over batches):
          - "entropy"
          - "long_ratio_8"
          - "long_ratio_16"
          - "long_ratio" (back-compat; equals long_ratio_8)
          - "pearson_r"
        """
        out = {"entropy": [], "long_ratio_8": [], "long_ratio_16": [], "long_ratio": [], "pearson_r": []}
        for li in range(self.n_layers):
            ent = torch.stack(self.buffers[li].ent, 0).mean(0) if self.buffers[li].ent else torch.tensor([])
            lr8 = torch.stack(self.buffers[li].long_ratio_8, 0).mean(0) if self.buffers[li].long_ratio_8 else torch.tensor([])
            lr16= torch.stack(self.buffers[li].long_ratio_16, 0).mean(0) if self.buffers[li].long_ratio_16 else torch.tensor([])
            pr  = torch.stack(self.buffers[li].pearson, 0).mean(0) if self.buffers[li].pearson else torch.tensor([])

            out["entropy"].append(ent)        # [H]
            out["long_ratio_8"].append(lr8)   # [H]
            out["long_ratio_16"].append(lr16) # [H]
            out["long_ratio"].append(lr8)     # back-compat: same as cutoff=8
            out["pearson_r"].append(pr)       # [H]
        return out