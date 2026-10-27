"""Marginals of the coefficients of the critically decimated Db4 DWT, data vs generated samples.

The Db4 arm imposes its scalar statistics on these coefficients (one marginal per scale, d coefficients
in total), so this is the table that says how well each model, Morlet or Db4, reproduces them. Same
quantities as run_comparison.channel_table (1 = perfect): m_p = E|c|^p ratios and quantile ratios of |c|,
one row per scale (j0 = finest, last row = the approximation block, i.e. the k = 0 mode).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes.ortho_wavelet.periodic_dwt import PeriodicDWT                  # noqa: E402


def dwt_coeffs(x, dwt):
    """list over scales of the flattened |coefficients| of x (B, 1, d) or (B, d)."""
    c = dwt.analysis(x.reshape(x.shape[0], 1, -1).double())
    return [c[:, j, :dwt.sizes[j]].reshape(-1).abs().numpy() for j in range(dwt.n_channels)]


def dwt_scale_table(x_data, xs, order=4, J=8, moments=(2, 4, 6), quantiles=(0.99, 0.999, 0.9999)):
    """{model: DataFrame (scales x ratios)} of generated / data. xs: {name: samples (B, d) or (B, 1, d)}."""
    d = x_data.shape[-1]
    dwt = PeriodicDWT(order, J, d, device='cpu')
    ref = dwt_coeffs(x_data.cpu(), dwt)
    out = {}
    for name, x in xs.items():
        gen = dwt_coeffs(x.cpu(), dwt)
        rows = []
        for j in range(dwt.n_channels):
            row = {'N per series': int(dwt.sizes[j])}
            for p in moments:
                row[f'm{p}'] = (gen[j] ** p).mean() / (ref[j] ** p).mean()
            for q in quantiles:
                row[f'q{q:g}'] = np.quantile(gen[j], q) / np.quantile(ref[j], q)
            rows.append(row)
        out[name] = pd.DataFrame(rows, index=[f'j{j}' if j < J else 'approx' for j in range(dwt.n_channels)])
    return out
