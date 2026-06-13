"""Generation point metrics: fidelity, discriminative_score, correlation, kl_divergence.

Input shapes:
  real:      (N, C, T) or (C, T)  -- real time series samples
  generated: (M, C, T) or (C, T)  -- generated time series samples (M may != N)
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


def fidelity(real, generated):
    """Per-feature Wasserstein-1 distance (averaged across features).

    Computes 1D Wasserstein distance between sorted real and generated values
    for each feature, then averages.
    """
    r, g = _validate(real, generated)
    N, C, T = r.shape
    M = g.shape[0]

    r_flat = r.reshape(N, C * T)  # (N, C*T)
    g_flat = g.reshape(M, C * T)  # (M, C*T)

    r_sorted, _ = r_flat.sort(dim=0)
    g_sorted, _ = g_flat.sort(dim=0)

    min_len = min(N, M)
    r_sorted = r_sorted[:min_len]
    g_sorted = g_sorted[:min_len]

    return torch.mean(torch.abs(r_sorted - g_sorted))


def discriminative_score(real, generated):
    """Discriminative score: |accuracy - 0.5| using a simple linear classifier.

    Trains a linear classifier to distinguish real from generated.
    A score of 0 means perfect indistinguishability.
    """
    r, g = _validate(real, generated)
    N, C, T = r.shape
    M = g.shape[0]

    r_flat = r.reshape(N, C * T)
    g_flat = g.reshape(M, C * T)

    X = torch.cat([r_flat, g_flat], dim=0)
    y = torch.cat([torch.zeros(N, device=r.device), torch.ones(M, device=r.device)])

    # Simple logistic regression via gradient descent
    perm = torch.randperm(len(y), device=r.device)
    X, y = X[perm], y[perm]

    n_train = int(0.8 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    w = torch.zeros(X.shape[1], device=r.device, requires_grad=True)
    b = torch.zeros(1, device=r.device, requires_grad=True)

    optim = torch.optim.Adam([w, b], lr=0.01)
    for _ in range(100):
        logits = X_train @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        logits = X_test @ w + b
        preds = (logits > 0).float()
        acc = (preds == y_test).float().mean()

    return torch.abs(acc - 0.5)


def correlation(real, generated, n_subsamples=5):
    """Cross-correlation preservation (Diffusion-TS style).

    Compares lag-wise autocorrelation structure between real and generated.
    """
    r, g = _validate(real, generated)
    N, C, T = r.shape
    M = g.shape[0]

    def _autocorr(x):
        """Compute autocorrelation for each feature, averaged over samples."""
        B, C_, T_ = x.shape
        acfs = []
        for c in range(C_):
            x_c = x[:, c, :]  # (B, T_)
            x_c = x_c - x_c.mean(dim=1, keepdim=True)
            var = (x_c ** 2).mean(dim=1, keepdim=True).clamp(min=1e-8)
            for lag in range(1, T_):
                cov = (x_c[:, :-lag] * x_c[:, lag:]).mean()
                acf = cov / var.mean()
                acfs.append(acf)
        return torch.stack(acfs)

    real_acf = _autocorr(r)
    gen_acf = _autocorr(g)

    return torch.abs(real_acf - gen_acf).mean()


def kl_divergence(real, generated, n_bins=50):
    """Per-feature KL divergence via histogram comparison.

    KL(real || generated), averaged across features.
    """
    r, g = _validate(real, generated)
    N, C, T = r.shape
    M = g.shape[0]

    r_flat = r.reshape(N * T, C)
    g_flat = g.reshape(M * T, C)

    total_kl = torch.tensor(0.0, device=r.device)
    for c in range(C):
        r_vals = r_flat[:, c]
        g_vals = g_flat[:, c]

        all_vals = torch.cat([r_vals, g_vals])
        bins = torch.linspace(all_vals.min(), all_vals.max(), n_bins + 1, device=r.device)

        r_hist = torch.histc(r_vals, bins=n_bins, min=bins[0], max=bins[-1])
        g_hist = torch.histc(g_vals, bins=n_bins, min=bins[0], max=bins[-1])

        r_hist = r_hist / r_hist.sum().clamp(min=1) + 1e-8
        g_hist = g_hist / g_hist.sum().clamp(min=1) + 1e-8

        total_kl = total_kl + (r_hist * torch.log(r_hist / g_hist)).sum()

    return total_kl / C


POINT_METRICS = ["Fidelity", "DiscriminativeScore", "Correlation", "KLDivergence"]

POINT_METRIC_FUNCS = {
    "Fidelity": fidelity,
    "DiscriminativeScore": discriminative_score,
    "Correlation": correlation,
    "KLDivergence": kl_divergence,
}
