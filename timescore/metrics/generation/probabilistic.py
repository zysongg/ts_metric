"""Generation probabilistic metrics: MMD, JS divergence, log_likelihood.

Input shapes:
  real:      (N, C, T) or (C, T)
  generated: (M, C, T) or (C, T)
"""

import torch
from ...utils import ensure_3d, clean_tensor


def _validate(real, generated):
    real = ensure_3d(clean_tensor(real, "real"), "real")
    generated = ensure_3d(clean_tensor(generated, "generated"), "generated")
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError(
            f"Feature/time mismatch: real {real.shape} vs generated {generated.shape}. "
            f"(C, T) must match."
        )
    return real, generated


def _rbf_kernel(X, Y, sigma=None):
    """Compute RBF kernel matrix between X and Y.

    X: (N, D), Y: (M, D) -> (N, M)
    """
    if sigma is None:
        sigma = torch.median(torch.cdist(X, Y))
        sigma = sigma.clamp(min=1e-5)

    XX = (X ** 2).sum(dim=1, keepdim=True)
    YY = (Y ** 2).sum(dim=1, keepdim=True)
    dist = XX + YY.T - 2 * X @ Y.T
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd(real, generated, sigma=None):
    """Maximum Mean Discrepancy with RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    r, g = _validate(real, generated)
    N, C, T = r.shape
    M = g.shape[0]

    r_flat = r.reshape(N, C * T)
    g_flat = g.reshape(M, C * T)

    k_rr = _rbf_kernel(r_flat, r_flat, sigma)
    k_gg = _rbf_kernel(g_flat, g_flat, sigma)
    k_rg = _rbf_kernel(r_flat, g_flat, sigma)

    mmd_sq = k_rr.mean() + k_gg.mean() - 2 * k_rg.mean()
    return torch.sqrt(mmd_sq.clamp(min=0))


def js_divergence(real, generated, n_bins=50):
    """Jensen-Shannon divergence (symmetric, bounded [0, 1]).

    JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = (P+Q)/2
    Computed per-feature and averaged.
    """
    r, g = _validate(real, generated)
    N, C, T = r.shape
    M = g.shape[0]

    r_flat = r.reshape(N * T, C)
    g_flat = g.reshape(M * T, C)

    total_js = torch.tensor(0.0, device=r.device)
    for c in range(C):
        r_vals = r_flat[:, c]
        g_vals = g_flat[:, c]

        all_vals = torch.cat([r_vals, g_vals])
        bins = torch.linspace(all_vals.min(), all_vals.max(), n_bins + 1, device=r.device)

        p = torch.histc(r_vals, bins=n_bins, min=bins[0], max=bins[-1])
        q = torch.histc(g_vals, bins=n_bins, min=bins[0], max=bins[-1])

        p = p / p.sum().clamp(min=1) + 1e-8
        q = q / q.sum().clamp(min=1) + 1e-8
        m = 0.5 * (p + q)

        kl_pm = (p * torch.log(p / m)).sum()
        kl_qm = (q * torch.log(q / m)).sum()
        total_js = total_js + 0.5 * (kl_pm + kl_qm)

    return total_js / C


def log_likelihood(real, generated):
    """Log-likelihood of real data under a Gaussian fitted to generated data.

    Fits per-feature Gaussian(mean, var) from generated, evaluates on real.
    """
    r, g = _validate(real, generated)
    N, C, T = r.shape
    M = g.shape[0]

    g_flat = g.reshape(M, C * T)
    mean = g_flat.mean(dim=0).reshape(1, C, T)
    var = g_flat.var(dim=0).reshape(1, C, T).clamp(min=1e-6)

    ll = -0.5 * (torch.log(2 * torch.pi * var) + (r - mean) ** 2 / var)
    return ll.mean()


PROB_METRICS = ["MMD", "JSDivergence", "LogLikelihood"]

PROB_METRIC_FUNCS = {
    "MMD": mmd,
    "JSDivergence": js_divergence,
    "LogLikelihood": log_likelihood,
}
