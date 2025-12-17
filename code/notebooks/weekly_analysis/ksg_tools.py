from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Literal

# Mutual information and entropy estimators from NPEET (KSG-style kNN estimators).
# Installed via: pip install git+https://github.com/gregversteeg/NPEET.git
from npeet import entropy_estimators as ee

# Public API for this module.
__all__ = [
    "ksg_mi_estimator_I",
    "moving_block_indices",
    "bootstrap_mi_mbb",
    "mi_percentile_ci_mbb",
    "redundancy_matrix",
    "rolling_redundancy",
    "redundancy_for_opt",
    "bits",
]


# Core mutual information estimator.
# This is a thin wrapper around NPEET’s implementation of KSG Estimator I.

def ksg_mi_estimator_I(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
    tau: Optional[float] = None,
    jitter_if_ties: bool = True,
    return_bits: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    KSG mutual information estimator (Estimator I), backed by NPEET.

    The estimator operates in log base e (nats) by default, with an option
    to convert to bits. A very small jitter can be added to break exact
    duplicate points before the kNN search.
    """
    x = np.asarray(x, dtype=float).copy().ravel()
    y = np.asarray(y, dtype=float).copy().ravel()

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    N = x.shape[0]
    if N <= k:
        raise ValueError(f"N={N} must be > k={k}")

    # Add a tiny amount of noise if there are exact duplicate observations.
    if jitter_if_ties:
        Z = np.column_stack([x, y])
        if np.unique(Z, axis=0).shape[0] < Z.shape[0]:
            if rng is None:
                rng = np.random.default_rng(12345)
            scale = np.nanstd(Z, axis=0)
            s = float(np.nanmean(scale)) if np.isfinite(scale).all() and np.nanmean(scale) > 0 else 1.0
            noise = max(1e-12, 1e-9 * s)
            x = x + rng.normal(0.0, noise, size=N)
            y = y + rng.normal(0.0, noise, size=N)

    I_nats = float(ee.mi(x.reshape(-1, 1), y.reshape(-1, 1), k=k))
    if return_bits:
        return I_nats / np.log(2.0)
    return I_nats


# Moving-block bootstrap utilities for time-series MI estimation.

def moving_block_indices(N: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate resampling indices by stitching together random contiguous blocks.

    Blocks are sampled with replacement; the final block is truncated so the
    total length is exactly N.
    """
    if block_len < 1 or block_len > N:
        raise ValueError("block_len must be between 1 and N")
    starts = rng.integers(0, max(1, N - block_len + 1), size=int(np.ceil(N / block_len)))
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)
    return idx[:N]


def bootstrap_mi_mbb(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
    B: int = 500,
    block_len: int = 12,
    seed: int = 123,
    return_bits: bool = False,
) -> np.ndarray:
    """
    Moving-block bootstrap for mutual information.

    Returns an array of B bootstrap MI estimates, each computed on a
    block-resampled version of the original series.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    N = len(x)
    if N <= k:
        raise ValueError(f"N={N} must be > k={k}")

    rng = np.random.default_rng(seed)
    out = np.empty(B, dtype=float)
    for b in range(B):
        idx = moving_block_indices(N, block_len, rng)
        out[b] = ksg_mi_estimator_I(
            x[idx], y[idx],
            k=k,
            return_bits=return_bits,
            jitter_if_ties=True,
            rng=rng,
        )
    return out


def mi_percentile_ci_mbb(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
    B: int = 500,
    block_len: int = 12,
    alpha: float = 0.05,
    seed: int = 123,
    return_bits: bool = False,
) -> Tuple[float, float, float]:
    """
    Point estimate and percentile confidence interval for MI
    using a moving-block bootstrap.
    """
    point = ksg_mi_estimator_I(x, y, k=k, return_bits=return_bits)
    boot = bootstrap_mi_mbb(
        x, y,
        k=k,
        B=B,
        block_len=block_len,
        seed=seed,
        return_bits=return_bits,
    )
    lo, hi = np.percentile(boot, [100 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return point, float(lo), float(hi)


# Small helpers used throughout the redundancy calculations.

def bits(x_nats: float) -> float:
    """
    Convert mutual information from nats to bits.
    """
    return x_nats / np.log(2.0)


def _ensure_datetime_index(df: pd.DataFrame) -> None:
    """
    Guardrail to ensure time-series inputs are indexed by dates.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")


def _entropy_estimate(x: np.ndarray, k: int = 4, return_bits: bool = True) -> float:
    """
    Differential entropy estimate via NPEET.

    Returns entropy in nats by default, or bits if requested.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    h_nats = float(ee.entropy(x, k=k))
    return h_nats / np.log(2.0) if return_bits else h_nats


def _normalize_mi_matrix(
    R: np.ndarray,
    H: np.ndarray,
    how: Literal["sqrt", "min"],
) -> np.ndarray:
    """
    Normalize a mutual information matrix using the diagonal entropies.

    sqrt: divide by sqrt(H_i * H_j)
    min:  divide by min(H_i, H_j)
    """
    if how == "sqrt":
        den = np.sqrt(np.outer(H, H))
        with np.errstate(invalid="ignore", divide="ignore"):
            return R / den
    elif how == "min":
        M = np.minimum.outer(H, H)
        with np.errstate(invalid="ignore", divide="ignore"):
            return R / M
    else:
        raise ValueError("how must be 'sqrt' or 'min'")


# Static (single-window) redundancy matrix with optional shuffle-based debiasing.

def redundancy_matrix(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    use_log_returns: bool = True,
    returns_are_log: bool = False,
    standardize: bool = True,
    min_obs: int = 60,
    k: int = 4,
    return_bits: bool = True,
    diag: Literal["entropy", "nan", "one"] = "entropy",
    clamp_negative: bool = False,
    normalize: Optional[Literal["sqrt", "min"]] = None,
    shuffle_debias: bool = False,
    ns_shuffles: int = 20,
    shuffle_seed: int = 12345,
    debias_only_if_leq: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a pairwise redundancy matrix using KSG mutual information.

    Each entry is computed using pairwise overlap only. When shuffle_debias
    is enabled, small or near-zero MI values are corrected using a shuffle-null
    mean and projected back to nonnegative values.
    """
    _ensure_datetime_index(df)

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]
    if df.empty:
        raise ValueError("No data after applying date filters.")

    if use_log_returns:
        rets = np.log(df / df.shift(1))
    else:
        rets = df.copy()
        if not returns_are_log:
            rets = np.log1p(rets)

    used = rets.copy()
    if standardize:
        used = (used - used.mean()) / used.std(ddof=0)

    tickers = used.columns.tolist()
    n = len(tickers)
    R = np.full((n, n), np.nan, dtype=float)

    rng = np.random.default_rng(shuffle_seed) if shuffle_debias else None

    for i in range(n):
        xi = used.iloc[:, i]
        for j in range(i + 1, n):
            yj = used.iloc[:, j]
            mask = xi.notna() & yj.notna()
            n_pair = int(mask.sum())
            if n_pair < min_obs:
                continue

            x = xi[mask].to_numpy()
            y = yj[mask].to_numpy()

            mi = ksg_mi_estimator_I(x, y, k=k, return_bits=return_bits)

            if shuffle_debias and np.isfinite(mi) and mi <= debias_only_if_leq:
                xs = (x - np.nanmean(x)) / (np.nanstd(x, ddof=0) or 1.0)
                ys = (y - np.nanmean(y)) / (np.nanstd(y, ddof=0) or 1.0)

                mi_obs = ksg_mi_estimator_I(xs, ys, k=k, return_bits=return_bits)

                x_clone = xs.copy()
                outs = []
                for _ in range(ns_shuffles):
                    rng.shuffle(x_clone)
                    outs.append(ksg_mi_estimator_I(x_clone, ys, k=k, return_bits=return_bits))
                mi_null = float(np.mean(outs)) if outs else 0.0

                mi = max(0.0, mi_obs - mi_null)

            if clamp_negative and mi < 0.0:
                mi = 0.0

            R[i, j] = R[j, i] = mi

    H = np.full(n, np.nan, dtype=float)
    if diag == "entropy" or normalize is not None:
        for i in range(n):
            xi = used.iloc[:, i].dropna().to_numpy()
            if len(xi) >= max(k + 1, 2):
                H[i] = _entropy_estimate(xi, k=k, return_bits=return_bits)

        if diag == "entropy":
            for i in range(n):
                R[i, i] = H[i]
        elif diag == "nan":
            np.fill_diagonal(R, np.nan)
        elif diag == "one":
            np.fill_diagonal(R, 1.0)
        else:
            raise ValueError("diag must be one of {'entropy','nan','one'}")
    else:
        if diag == "nan":
            np.fill_diagonal(R, np.nan)
        elif diag == "one":
            np.fill_diagonal(R, 1.0)
        elif diag == "entropy":
            pass
        else:
            raise ValueError("diag must be one of {'entropy','nan','one'}")

    if normalize is not None:
        if not np.any(np.isfinite(H)):
            raise ValueError("Cannot normalize without valid entropies on the diagonal.")
        N = _normalize_mi_matrix(R, H, how=normalize)

        if diag == "entropy":
            for i in range(n):
                N[i, i] = 1.0 if np.isfinite(H[i]) and H[i] > 0 else np.nan
        elif diag == "nan":
            np.fill_diagonal(N, np.nan)
        elif diag == "one":
            np.fill_diagonal(N, 1.0)

        R = N

    return pd.DataFrame(R, index=tickers, columns=tickers), used


# Rolling-window redundancy matrices for time-varying analysis.

def rolling_redundancy(
    df: pd.DataFrame,
    window: int,
    step: int = 1,
    *,
    use_log_returns: bool = True,
    returns_are_log: bool = False,
    standardize: bool = True,
    min_obs: int = 60,
    k: int = 4,
    return_bits: bool = True,
    diag: Literal["entropy", "nan", "one"] = "entropy",
    clamp_negative: bool = False,
    normalize: Optional[Literal["sqrt", "min"]] = None,
) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
    """
    Compute redundancy matrices over a rolling window.
    """
    _ensure_datetime_index(df)

    if use_log_returns:
        base = np.log(df / df.shift(1)).dropna(how="all")
    else:
        base = df.copy()
        if not returns_are_log:
            base = np.log1p(base)
    base = base.dropna(axis=0)

    out: List[Tuple[pd.Timestamp, pd.DataFrame]] = []

    for end_ix in range(window, len(base) + 1, step):
        sl = base.iloc[end_ix - window : end_ix].dropna(axis=0)
        if len(sl) < min_obs:
            continue
        if standardize:
            sl = (sl - sl.mean()) / sl.std(ddof=0)

        tickers = sl.columns.tolist()
        n = len(tickers)
        R = np.full((n, n), np.nan, dtype=float)
        data = sl.to_numpy()

        for i in range(n):
            xi = data[:, i]
            for j in range(i + 1, n):
                yj = data[:, j]
                m = min(len(xi), len(yj))
                if m < min_obs:
                    continue
                mi = ksg_mi_estimator_I(xi[:m], yj[:m], k=k, return_bits=return_bits)
                if clamp_negative and mi < 0.0:
                    mi = 0.0
                R[i, j] = R[j, i] = mi

        H = np.full(n, np.nan, dtype=float)
        if diag == "entropy" or normalize is not None:
            for i in range(n):
                xi = data[:, i]
                if len(xi) >= max(k + 1, 2):
                    H[i] = _entropy_estimate(xi, k=k, return_bits=return_bits)

            if diag == "entropy":
                for i in range(n):
                    R[i, i] = H[i]
            elif diag == "nan":
                np.fill_diagonal(R, np.nan)
            elif diag == "one":
                np.fill_diagonal(R, 1.0)
            else:
                raise ValueError("diag must be one of {'entropy','nan','one'}")
        else:
            if diag == "nan":
                np.fill_diagonal(R, np.nan)
            elif diag == "one":
                np.fill_diagonal(R, 1.0)
            elif diag == "entropy":
                pass
            else:
                raise ValueError("diag must be one of {'entropy','nan','one'}")

        if normalize is not None:
            if not np.any(np.isfinite(H)):
                continue
            N = _normalize_mi_matrix(R, H, how=normalize)

            if diag == "entropy":
                for i in range(n):
                    N[i, i] = 1.0 if np.isfinite(H[i]) and H[i] > 0 else np.nan
            elif diag == "nan":
                np.fill_diagonal(N, np.nan)
            elif diag == "one":
                np.fill_diagonal(N, 1.0)

            R = N

        out.append((sl.index[-1], pd.DataFrame(R, index=tickers, columns=tickers)))

    return out


# Helper for portfolio optimization: normalized, nonnegative redundancy matrix.

def redundancy_for_opt(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    use_log_returns: bool = True,
    returns_are_log: bool = False,
    standardize: bool = True,
    min_obs: int = 60,
    k: int = 4,
    normalize: Literal["sqrt", "min"] = "sqrt",
    clamp_negative: bool = False,
    shrink_eps: Optional[float] = 0.0,
    shuffle_debias: bool = True,
    ns_shuffles: int = 20,
    shuffle_seed: int = 12345,
    debias_only_if_leq: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construct an optimizer-ready redundancy matrix.

    The output is normalized, nonnegative, and has a zero diagonal,
    which makes it convenient for quadratic objectives.
    """
    R_raw, used = redundancy_matrix(
        df, start, end,
        use_log_returns=use_log_returns,
        returns_are_log=returns_are_log,
        standardize=standardize,
        min_obs=min_obs,
        k=k,
        return_bits=True,
        diag="entropy",
        clamp_negative=clamp_negative,
        normalize=None,
        shuffle_debias=shuffle_debias,
        ns_shuffles=ns_shuffles,
        shuffle_seed=shuffle_seed,
        debias_only_if_leq=debias_only_if_leq,
    )

    # Entropy vector (bits)
    H = pd.Series(np.diag(R_raw.values), index=R_raw.index, name="entropy_bits")

    # Normalize to NMI
    Hv = H.values
    N = _normalize_mi_matrix(R_raw.values, Hv, how=normalize)

    # Diagonal policy for optimization: diag = 0
    np.fill_diagonal(N, 0.0)

    # Small-entry shrinkage (gentle denoising; does not clamp to 0)
    if shrink_eps is not None and shrink_eps > 0.0:
        mask_small = (N < shrink_eps) & (N > 0)
        N[mask_small] *= 0.5

    NMI_opt = pd.DataFrame(N, index=R_raw.index, columns=R_raw.columns)
    return NMI_opt, H, R_raw
