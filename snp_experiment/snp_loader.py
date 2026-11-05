"""Daily S&P 500 loader for MGD: the whole series as ONE sample (MGD paper, Sec. 6.3).

Source: snp_WSJ_08_02_2024.pkl (Morel et al., scattering_spectra repo), a DataFrame indexed by Date with
Open/High/Low/Close, stored newest-first. Path: env var SNP_PKL_PATH, else data/data_files/ of this repo if the
file is there, else the local clone of scattering_spectra.

Only Close is used (Open == Close on ~2000 rows of 2000-2007, High/Low unreliable there).
X(u) = log P(u) - log P(u-1): 6062 daily log-returns, consecutive trading days = consecutive samples (weekends,
holidays and closures ignored). Standardized once, globally (mean 0, std 1). No windows: d = 6062, the moments
are averages over u of the whole series.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

_REPO_COPY = Path(__file__).resolve().parents[1] / "data/data_files/snp_WSJ_08_02_2024.pkl"
_MOREL_CLONE = Path.home() / "phd/scattering_spectra/scatspectra/data/snp_WSJ_08_02_2024.pkl"
DEFAULT_PKL = _REPO_COPY if _REPO_COPY.exists() else _MOREL_CLONE   # repo copy (what JZ sees) first


def load_snp_close(path=None):
    """Return (close, dates): Close prices, ascending in time, and the DatetimeIndex."""
    path = Path(path or os.environ.get("SNP_PKL_PATH", DEFAULT_PKL))
    df = pd.read_pickle(path).sort_index()          # file is newest-first: reversed time flips asymmetry stats
    assert not df.isna().any().any(), "NaN in the S&P table"
    assert not df.index.duplicated().any(), "duplicate dates"
    assert df.index.is_monotonic_increasing and df.index.is_unique, "index not strictly increasing after sort"
    close = df["Close"].to_numpy(dtype=np.float64)
    assert (close > 0).all()
    return close, df.index


def load_snp_returns(path=None):
    """Return (x, raw, dates): x = standardized daily log-returns as a float32 tensor (1, 1, d), raw = the
    unstandardized float64 returns (numpy), dates = the date of each return (the day of P(u))."""
    close, dates = load_snp_close(path)
    raw = np.diff(np.log(close))
    x = (raw - raw.mean()) / raw.std()
    return torch.from_numpy(x.astype(np.float32))[None, None, :], raw, dates[1:]


if __name__ == "__main__":
    x, raw, dates = load_snp_returns()
    print("d = %d daily log-returns, %s -> %s" % (raw.size, dates[0].date(), dates[-1].date()))
    print("raw returns: std %.4f  skew %.3f  excess kurtosis %.2f" % (raw.std(), stats.skew(raw), stats.kurtosis(raw)))
    print("standardized x: shape %s dtype %s mean %.2e std %.4f" % (tuple(x.shape), x.dtype, x.mean(), x.std()))
