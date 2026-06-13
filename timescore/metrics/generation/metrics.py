"""Time series generation evaluation metrics (TSGBench-style).

All metrics compare two sets of time series samples:
  real:      (N, C, T) — N real samples
  generated: (M, C, T) — M generated samples

Categories:
  Feature-based:  MDD, ACD, SD, KD
  Distance-based: ED, DTW
  Model-based:    DS, PS, C_FID
"""

import torch
import torch.nn as nn
import numpy as np

from ...utils import ensure_3d, clean_tensor


def _validate(real, generated):
    real = ensure_3d(clean_tensor(real, "real"), "real")
    generated = ensure_3d(clean_tensor(generated, "generated"), "generated")
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError(
            f"Feature/time mismatch: real {real.shape} vs generated {generated.shape}"
        )
    return real, generated


# ==================== Feature-based metrics ====================

def _histogram_loss(x_real, x_fake, n_bins=50):
    """Marginal distribution distance via histogram comparison.

    x_real, x_fake: (N, T, C) — note: time-first convention internally.
    Returns scalar loss.
    """
    N_r, T, C = x_real.shape
    N_f = x_fake.shape[0]
    total_loss = 0.0
    count = 0

    for c in range(C):
        for t in range(T):
            r_vals = x_real[:, t, c]
            f_vals = x_fake[:, t, c]

            all_vals = torch.cat([r_vals, f_vals])
            a, b = all_vals.min().item(), all_vals.max().item()
            b = b + 1e-5 if b == a else b

            bins = torch.linspace(a, b, n_bins + 1, device=x_real.device)
            delta = bins[1] - bins[0]

            r_hist = torch.histc(r_vals, bins=n_bins, min=a, max=b).float()
            f_hist = torch.histc(f_vals, bins=n_bins, min=a, max=b).float()

            r_density = r_hist / delta / N_r
            f_density = f_hist / delta / N_f

            total_loss += torch.abs(r_density - f_density).mean().item()
            count += 1

    return total_loss / max(count, 1)


def mdd(real, generated, n_bins=50):
    """Marginal Distribution Distance (TSGBench: MDD).

    Compares per-feature per-timestep marginal distributions via histograms.
    """
    r, g = _validate(real, generated)
    # Convert (N, C, T) -> (N, T, C) for histogram computation
    r_tc = r.permute(0, 2, 1)
    g_tc = g.permute(0, 2, 1)
    return torch.tensor(_histogram_loss(r_tc, g_tc, n_bins), device=r.device)


def _acf(x, max_lag):
    """Compute auto-correlation function. x: (N, T, C) -> (max_lag, C)."""
    x = x - x.mean(dim=(0, 1), keepdim=True)
    std = x.var(dim=(0, 1), unbiased=False)
    acf_list = []
    for i in range(max_lag):
        if i == 0:
            y = x ** 2
        else:
            y = x[:, i:] * x[:, :-i]
        acf_i = y.mean(dim=(0, 1)) / std.clamp(min=1e-8)
        acf_list.append(acf_i)
    return torch.stack(acf_list)  # (max_lag, C)


def acd(real, generated, max_lag=None):
    """Auto-Correlation Distance (TSGBench: ACD).

    L2 norm of ACF difference between real and generated.
    """
    r, g = _validate(real, generated)
    r_tc = r.permute(0, 2, 1)  # (N, T, C)
    g_tc = g.permute(0, 2, 1)
    T = r_tc.shape[1]

    if max_lag is None:
        max_lag = min(64, T)

    acf_real = _acf(r_tc, max_lag)
    acf_fake = _acf(g_tc, max_lag)
    diff = acf_fake - acf_real
    return torch.sqrt((diff ** 2).sum(dim=0)).mean()


def _skewness(x):
    """Compute skewness. x: (N, T, C) -> (C,)."""
    x = x - x.mean(dim=(0, 1), keepdim=True)
    x_3 = (x ** 3).mean(dim=(0, 1))
    x_std = x.std(dim=(0, 1), unbiased=True).clamp(min=1e-8)
    return x_3 / (x_std ** 3)


def sd(real, generated):
    """Skewness Difference (TSGBench: SD).

    Mean absolute difference in skewness between real and generated.
    """
    r, g = _validate(real, generated)
    r_tc = r.permute(0, 2, 1)
    g_tc = g.permute(0, 2, 1)
    skew_real = _skewness(r_tc)
    skew_fake = _skewness(g_tc)
    return torch.abs(skew_fake - skew_real).mean()


def _kurtosis(x):
    """Compute excess kurtosis. x: (N, T, C) -> (C,)."""
    x = x - x.mean(dim=(0, 1), keepdim=True)
    x_4 = (x ** 4).mean(dim=(0, 1))
    x_var = x.var(dim=(0, 1), unbiased=False).clamp(min=1e-8)
    return x_4 / (x_var ** 2) - 3.0


def kd(real, generated):
    """Kurtosis Difference (TSGBench: KD).

    Mean absolute difference in excess kurtosis between real and generated.
    """
    r, g = _validate(real, generated)
    r_tc = r.permute(0, 2, 1)
    g_tc = g.permute(0, 2, 1)
    kurt_real = _kurtosis(r_tc)
    kurt_fake = _kurtosis(g_tc)
    return torch.abs(kurt_fake - kurt_real).mean()


# ==================== Distance-based metrics ====================

def ed(real, generated):
    """Euclidean Distance (TSGBench: ED).

    Average Euclidean distance between paired samples (first min(N,M) pairs).
    """
    r, g = _validate(real, generated)
    n_pairs = min(r.shape[0], g.shape[0])
    r_p = r[:n_pairs]  # (P, C, T)
    g_p = g[:n_pairs]

    # Per-sample distance averaged over features
    dists = torch.norm(r_p - g_p, dim=2).mean(dim=1)  # (P,)
    return dists.mean()


def dtw(real, generated):
    """Dynamic Time Warping Distance (TSGBench: DTW).

    Average DTW distance between paired samples.
    Requires: pip install dtaidistance
    """
    try:
        from dtaidistance.dtw_ndim import distance as multi_dtw_distance
    except ImportError:
        raise ImportError("dtaidistance is required for DTW. Install with: pip install dtaidistance")

    r, g = _validate(real, generated)
    n_pairs = min(r.shape[0], g.shape[0])

    # Convert to numpy (N, C, T) -> (N, T, C) for dtaidistance
    r_np = r[:n_pairs].permute(0, 2, 1).detach().cpu().numpy().astype(np.double)
    g_np = g[:n_pairs].permute(0, 2, 1).detach().cpu().numpy().astype(np.double)

    distances = []
    for i in range(n_pairs):
        d = multi_dtw_distance(r_np[i], g_np[i], use_c=True)
        distances.append(d)

    return torch.tensor(np.mean(distances), device=r.device, dtype=r.dtype)


# ==================== Model-based metrics ====================

class _GRUClassifier(nn.Module):
    """Simple GRU classifier for discriminative score."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze(-1)


class _GRUPredictor(nn.Module):
    """Simple GRU predictor for predictive score."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim or input_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def ds(real, generated, iterations=2000, hidden_dim=64, lr=0.01):
    """Discriminative Score (TSGBench: DS).

    Train a GRU classifier to distinguish real from generated.
    DS = |accuracy - 0.5|. Lower is better (0 = indistinguishable).

    Args:
        real: (N, C, T)
        generated: (M, C, T)
        iterations: training iterations
        hidden_dim: GRU hidden size
        lr: learning rate
    """
    r, g = _validate(real, generated)
    C, T = r.shape[1], r.shape[2]

    # Convert to (N, T, C) for GRU
    r_tc = r.permute(0, 2, 1)
    g_tc = g.permute(0, 2, 1)

    X = torch.cat([r_tc, g_tc], dim=0)
    y = torch.cat([torch.zeros(r_tc.shape[0]), torch.ones(g_tc.shape[0])], dim=0)

    perm = torch.randperm(len(y), device=r.device)
    X, y = X[perm], y[perm].to(r.device)

    n_train = int(0.8 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model = _GRUClassifier(C, hidden_dim).to(r.device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    batch_size = min(128, len(X_train))
    for _ in range(iterations):
        idx = torch.randint(0, len(X_train), (batch_size,), device=r.device)
        logits = model(X_train[idx])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train[idx])
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = (logits > 0).float()
        acc = (preds == y_test).float().mean()

    return torch.abs(acc - 0.5)


def ps(real, generated, iterations=2000, hidden_dim=64, lr=0.01):
    """Predictive Score (TSGBench: PS).

    Train a GRU on generated data to predict next step, evaluate on real data.
    PS = MAE on real test set. Lower is better.

    Args:
        real: (N, C, T)
        generated: (M, C, T)
        iterations: training iterations
        hidden_dim: GRU hidden size
        lr: learning rate
    """
    r, g = _validate(real, generated)
    C, T = r.shape[1], r.shape[2]

    # Convert to (N, T, C) for GRU
    r_tc = r.permute(0, 2, 1)
    g_tc = g.permute(0, 2, 1)

    # Train on generated: input is first T-1 steps, target is last step
    X_gen = g_tc[:, :-1, :]  # (M, T-1, C)
    y_gen = g_tc[:, -1, :]   # (M, C)

    model = _GRUPredictor(C, hidden_dim, C).to(r.device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    batch_size = min(128, len(X_gen))
    for _ in range(iterations):
        idx = torch.randint(0, len(X_gen), (batch_size,), device=r.device)
        pred = model(X_gen[idx])
        loss = nn.functional.mse_loss(pred, y_gen[idx])
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Evaluate on real
    X_real = r_tc[:, :-1, :]
    y_real = r_tc[:, -1, :]

    model.eval()
    with torch.no_grad():
        pred = model(X_real)
        mae = torch.abs(pred - y_real).mean()

    return mae


def c_fid(real, generated, ts2vec_model=None):
    """Context-FID (TSGBench: C-FID).

    Encode time series with ts2vec, then compute FID between representations.

    Args:
        real: (N, C, T)
        generated: (M, C, T)
        ts2vec_model: optional pre-trained ts2vec encoder. If None, uses a
            simple 1D-CNN encoder as fallback.
    """
    r, g = _validate(real, generated)

    if ts2vec_model is not None:
        # Use provided ts2vec model
        # ts2vec expects (N, T, C)
        r_tc = r.permute(0, 2, 1).detach().cpu().numpy()
        g_tc = g.permute(0, 2, 1).detach().cpu().numpy()
        ori_repr = ts2vec_model.encode(r_tc, encoding_window='full_series')
        gen_repr = ts2vec_model.encode(g_tc, encoding_window='full_series')
        ori_repr = torch.tensor(ori_repr, device=r.device, dtype=r.dtype)
        gen_repr = torch.tensor(gen_repr, device=r.device, dtype=r.dtype)
    else:
        # Fallback: use mean and std as simple representation
        ori_repr = torch.cat([r.mean(dim=2), r.std(dim=2)], dim=1)  # (N, 2C)
        gen_repr = torch.cat([g.mean(dim=2), g.std(dim=2)], dim=1)  # (M, 2C)

    return _compute_fid(ori_repr, gen_repr)


def _compute_fid(act1, act2):
    """Compute Frechet Inception Distance between two sets of representations."""
    act1 = act1.detach().cpu().numpy()
    act2 = act2.detach().cpu().numpy()

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return torch.tensor(float(fid))


def train_ts2vec(real_data, device=None, output_dims=100, batch_size=8, 
                 lr=0.001, max_train_length=3000, n_iters=None, verbose=False):
    """Train a TS2Vec encoder on real data for use with C-FID.
    
    This follows the TSGBench methodology: train TS2Vec on real training data,
    then use it to encode both real and generated data for FID computation.
    
    Args:
        real_data: Real time series data, shape (B, C, T) or (B, T, C)
        device: Device to use (default: cuda if available, else cpu)
        output_dims: Dimension of learned representations
        batch_size: Training batch size
        lr: Learning rate
        max_train_length: Maximum sequence length for training
        n_iters: Number of training iterations (default: auto based on data size)
        verbose: Whether to print training progress
    
    Returns:
        Trained TS2Vec model
    
    Example:
        >>> # Train on your training data
        >>> ts2vec_model = train_ts2vec(train_real_data)
        >>> 
        >>> # Use for C-FID on test data
        >>> cfid = c_fid(test_real, generated, ts2vec_model=ts2vec_model)
    """
    try:
        from .ts2vec import TS2Vec
    except ImportError:
        raise ImportError("TS2Vec not found. Please ensure ts2vec.py is available in the generation directory.")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Ensure (B, T, C) format for TS2Vec
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()
    
    # Detect format and convert to (B, T, C)
    if real_data.ndim == 3:
        B, dim1, dim2 = real_data.shape
        # Heuristic: if dim1 < dim2, assume (B, C, T) format
        if dim1 < dim2:
            real_data = real_data.transpose(0, 2, 1)  # (B, T, C)
    
    # Train TS2Vec
    model = TS2Vec(
        input_dims=real_data.shape[-1],
        output_dims=output_dims,
        device=device,
        batch_size=batch_size,
        lr=lr,
        max_train_length=max_train_length
    )
    
    model.fit(real_data, n_iters=n_iters, verbose=verbose)
    
    return model


METRICS = ["MDD", "ACD", "SD", "KD", "ED", "DTW", "DS", "PS", "C_FID"]

METRIC_FUNCS = {
    "MDD": mdd,
    "ACD": acd,
    "SD": sd,
    "KD": kd,
    "ED": ed,
    "DTW": dtw,
    "DS": ds,
    "PS": ps,
    "C_FID": c_fid,
}
