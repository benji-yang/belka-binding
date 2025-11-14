import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import configs.parameters as cf

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.0, reduction='mean', eps:float=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        with autocast(device_type=cf.DEVICE, enabled=False):
            logits = logits.float()
            targets = targets.float()
            # logits: (batch, num_classes), targets: same shape, 0/1
            # probs = torch.sigmoid(logits)
            # p_t = prob for true class
            # p_t = probs * targets + (1 - probs) * (1 - targets)
            # alpha_t = α for positives, (1-α) for negatives
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            # BCE with logits per-element
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            p_t = torch.exp(-bce_loss)
            focal_term = (1.0 - p_t).clamp(min=self.eps, max=1.0-self.eps)
            # focal scaling
            focal_factor = focal_term.pow(self.gamma)
            loss = alpha_t * focal_factor * bce_loss

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, margin=0.05, eps=1e-6, reduction='mean', detach_weights=True):
        super().__init__()
        self.gpos = gamma_pos
        self.gneg = gamma_neg
        self.m    = margin
        self.eps  = eps
        self.reduction = reduction
        self.detach_weights = detach_weights

    def forward(self, logits, targets):
        with autocast(device_type=cf.DEVICE, enabled=False):
            # probs
            p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
            pm = (p - self.m).clamp_min(0.0).clamp_max(1.0 - self.eps)  # probability shifting (negatives only)

            # base CE terms in probability space
            pos_ce = -torch.log(p)                 # y=1
            neg_ce = -torch.log1p(-pm)             # y=0, NOTE: uses 1 - p_m

            # focal-style modulating factors
            w_pos = (1.0 - p).pow(self.gpos)
            w_neg = pm.pow(self.gneg)

            if self.detach_weights:
                w_pos = w_pos.detach()
                w_neg = w_neg.detach()

            # assemble per-element loss
            loss = torch.where(
                targets.bool(),
                w_pos * pos_ce,
                w_neg * neg_ce
            )

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss


class MultiCenterSVDDLoss(nn.Module):
    """Calculates the SVDD loss for multiple classes, each with its own center."""
    def __init__(self, num_classes: int, embedding_dim: int, l2_normalize: bool = True, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        # Create a center for each class. These are learnable parameters.
        centers_init = torch.randn(num_classes, embedding_dim)
        if l2_normalize:
            centers_init = F.normalize(centers_init, p=2, dim=1, eps=eps)
        self.centers = nn.Parameter(centers_init)
        self.l2_normalize = l2_normalize
        self.eps = eps
        self.reduction = reduction

    @torch.no_grad()
    def renorm_centers_(self):
        """Project centers back to the unit sphere (optional to call after opt.step())."""
        self.centers.copy_(F.normalize(self.centers, p=2, dim=1, eps=self.eps))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor):
        """
        embeddings: [B, E]
        targets:    [B, C] (binary 0/1)
        returns: scalar loss (default 'mean')
        """
        targets = targets.float()

        if self.l2_normalize:
            # Normalize both sides
            emb = F.normalize(embeddings, p=2, dim=1, eps=self.eps)      # [B, E]
            cen = F.normalize(self.centers,   p=2, dim=1, eps=self.eps)  # [C, E]

            # dist^2 = 2 - 2 * cos; compute via matmul to avoid [B,C,E] broadcast
            sim = emb @ cen.t()                                          # [B, C]
            dist_matrix_sq = (2.0 - 2.0 * sim).clamp_min(0.0)            # numeric safety
        else:
            # Unnormalized Euclidean distance (original behavior)
            emb = embeddings.unsqueeze(1)                                 # [B,1,E]
            cen = self.centers.unsqueeze(0)                               # [1,C,E]
            diff = emb - cen                                              # [B,C,E]
            dist_matrix_sq = (diff * diff).sum(dim=2)                     # [B,C]

        # Mask to positives
        masked_dist = dist_matrix_sq * targets                            # [B,C]

        # Average per sample over its positives (avoid div by 0)
        num_pos = targets.sum(dim=1).clamp(min=1)
        per_sample = masked_dist.sum(dim=1) / num_pos                     # [B]

        if self.reduction == 'mean':
            return per_sample.mean()
        elif self.reduction == 'sum':
            return per_sample.sum()
        return per_sample


class CenterLoss(nn.Module):
    """
    Numerically-stable Center Loss with EMA center updates.
    Works for:
      - single-label: targets LongTensor [B]
      - multi-label:  targets FloatTensor [B,C] in {0,1}
    """
    def __init__(self, num_classes: int, feat_dim: int,
                 alpha: float = 0.2, eps: float = 1e-6, multi_label: bool = True,
                 max_feat_norm: float | None = None,  # optional: clip feature norms
                 l2_normalize_centers: bool = False   # optional: keep centers on unit sphere
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.multi_label = bool(multi_label)
        self.max_feat_norm = max_feat_norm
        self.l2_normalize_centers = l2_normalize_centers

        # Keep centers in FP32 for stability, regardless of model dtype
        self.register_buffer('centers', torch.empty(num_classes, feat_dim, dtype=torch.float32))
        nn.init.normal_(self.centers, mean=0.0, std=0.02)

    def forward(self, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        feats:   [B, E] (embedding vectors), any dtype (bf16/fp16/fp32)
        targets: single-label -> LongTensor [B]
                 multi-label  -> FloatTensor [B, C] with 0/1
        returns: scalar loss (FP32), cast back to feats.dtype for consistency with AMP
        """
        # --- sanitize & set dtypes (do all math in fp32) ---
        with autocast(device_type=cf.DEVICE, enabled=False):
            # with something safer:
            f32 = feats.to(torch.float32)
            # zero-out non-finite entries instead of inserting huge numbers
            finite = torch.isfinite(f32)
            if not finite.all():
                f32 = torch.where(finite, f32, torch.zeros_like(f32))
            centers = self.centers  # already fp32

            # Optional: clip feature norms to avoid huge squared distances
            if self.max_feat_norm is not None:
                nrm = f32.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
                scale = (self.max_feat_norm / nrm).clamp_max(1.0)
                f32 = f32 * scale

            if self.multi_label:
                B, E = f32.shape
                C = self.num_classes
                assert targets.shape == (B, C), "For multi_label=True, targets must be [B,C]"
                # boolean -> float mask (fp32) for stable matmul
                pos_mask = (targets > 0.5).to(torch.float32)  # [B,C]

                # squared distances to all centers
                # d2[i,j] = ||x_i - c_j||^2
                diff = f32.unsqueeze(1) - centers.unsqueeze(0)       # [B,C,E]
                d2 = (diff * diff).sum(dim=-1)                       # [B,C]

                num_pos = pos_mask.sum()
                # if no positives, return 0 without branching to NaN
                loss = 0.5 * (d2 * pos_mask).sum() / num_pos.clamp_min(1.0)

                # Update centers (EMA)
                self._update_centers_multi(f32, pos_mask)

            else:
                # single-label path
                assert targets.dim() == 1 and targets.dtype in (torch.long, torch.int64)
                c_y = centers.index_select(dim=0, index=targets)     # [B,E]
                diff = f32 - c_y
                loss = 0.5 * (diff * diff).sum(dim=1).mean()

                # Update centers (EMA, vectorized)
                self._update_centers_single(f32, targets)

            # Optionally keep centers on unit sphere (prevents drift)
            if self.l2_normalize_centers:
                with torch.no_grad():
                    self.centers.copy_(torch.nn.functional.normalize(self.centers, p=2, dim=1, eps=self.eps))

            # Return in original dtype to play nicely with AMP scalers
        return loss

    @torch.no_grad()
    def _update_centers_multi(self, feats_f32: torch.Tensor, pos_mask_f32: torch.Tensor):
        """
        Multi-label EMA update (vectorized).
        Δc_j = (Σ_i m_ij * x_i - count_j * c_j) / (eps + count_j)
        c_j <- c_j + α * Δc_j, only where count_j > 0.
        """
        centers = self.centers
        # counts per class
        counts = pos_mask_f32.sum(dim=0)                    # [C]
        # sums per class: [C,E]
        Xsum = pos_mask_f32.t().matmul(feats_f32)           # [C,B] @ [B,E] -> [C,E]
        counts_exp = counts.unsqueeze(1)
        denom = (self.eps + counts_exp)
        delta = (Xsum - counts_exp * centers) / denom
        # mask classes with zero count (avoid touching them)
        mask = (counts > 0).to(centers.dtype).unsqueeze(1)
        centers.add_(self.alpha * delta * mask)

    @torch.no_grad()
    def _update_centers_single(self, feats_f32: torch.Tensor, labels: torch.Tensor):
        """
        Single-label EMA update (vectorized using scatter-add).
        """
        C, E = self.num_classes, self.feat_dim
        centers = self.centers

        # counts per class
        counts = torch.bincount(labels, minlength=C).to(torch.float32)         # [C]
        # sums per class
        Xsum = torch.zeros_like(centers)                                       # [C,E]
        Xsum.index_add_(0, labels, feats_f32)                                  # scatter-add

        denom = (self.eps + counts).unsqueeze(1)                                # [C,1]
        delta = (Xsum - counts.unsqueeze(1) * centers) / denom                 # [C,E]
        mask = (counts > 0).to(centers.dtype).unsqueeze(1)
        centers.add_(self.alpha * delta * mask)