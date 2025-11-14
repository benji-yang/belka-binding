import math
import torch
import torch.nn as nn


class GaussianDistanceEmbedding(nn.Module):
    def __init__(self, K: int, d_min: float = 0.0, d_max: float = 10.0):
        """
        Args:
          K       : number of Gaussian kernels
          d_min   : minimum distance to initialize μ
          d_max   : maximum distance to initialize μ
        """
        super().__init__()
        # initialize μᵏ to be K points evenly spaced in [d_min, d_max]
        mu_init = torch.linspace(d_min, d_max, K)
        self.mu = nn.Parameter(mu_init)             # shape (K,)
        self.softplus = nn.Softplus()
        
        # we parameterize σ via a softplus for stability / positivity
        # start with all log_sigma = 0 → σᵏ = softplus(0) ≈ 0.693
        self.log_sigma = nn.Parameter(torch.zeros(K))

    def forward(self, edge_index: torch.Tensor, pos_matrix: torch.Tensor) -> torch.Tensor:
        """
        distances: tensor of shape (E,) or (B, E, ...) containing scalar d_ij = ||r_i - r_j||
        returns:   tensor of shape distances.shape + (K,)
                   where [:, k] = ψ^k_ij = - 1/√(2π σᵏ) * exp( -½ (d_ij - μᵏ)² / σᵏ )
        """
        i, j = edge_index
        d_ij = (pos_matrix[i] - pos_matrix[j]).norm(dim=1)
        # ensure σᵏ > 0
        sigma = self.softplus(self.log_sigma)            # (K,)

        # bring distances to shape (..., 1) so we can broadcast against (K,)
        d = d_ij.unsqueeze(-1)                   # (..., 1)

        # squared diff
        diff2 = (d - self.mu)**2                       # (..., K)

        # Gaussian kernel prefactor: 1 / sqrt(2π σᵏ)
        # note the negative sign in your formula
        prefactor = -1.0 / torch.sqrt(2 * math.pi * sigma)  # (K,)

        # exponent: exp( -½ * diff2 / σᵏ )
        exponent  = torch.exp(-0.5 * diff2 / sigma)    # (..., K)

        return prefactor * exponent