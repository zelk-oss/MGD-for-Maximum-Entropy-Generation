"""
Light loader for the Modane jet data (`Vit_Modane95.mat`).

The file holds a single continuous velocity record, `Vit`, shape (25803816, 1),
float64 (values at float32 resolution). It is the only data `run_jet_SDE.py`
trains on since commit 177dedd (Jul 17); the helium-jet loader in
`data/data_loader.py` is legacy.

Data audit (jets/data_audit/audit_modane.py, notes/jet_data_audit_1027/):
no NaN, no clipping, no stuck-probe zones -> nothing is excluded, the full
record is used.

    from jet_data import load_modane_raw, load_modane_windows, describe_modane

    v = load_modane_raw()                                  # (N,) float64, m/s (TODO: confirm units)
    x1, starts = load_modane_windows(window_len=1024, target_len=256, n=5000)
    describe_modane()
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# `codes` modules import each other as top-level packages (e.g. `potentials`).
for _p in (PROJECT_ROOT, PROJECT_ROOT / 'codes'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
MODANE_PATH = PROJECT_ROOT / 'data' / 'data_files' / 'helium_jets' / 'Vit_Modane95.mat'

# TODO: fill from the data README (deferred 2026-10-27).
MODANE_META = {
    'facility': None,
    'rlambda': None,
    'dt_s': None,
    'units': None,
}

# [start, stop) sample intervals never used for windows. Empty: the 10-27 audit
# found no bad zone.
EXCLUDED_INTERVALS: list[tuple[int, int, str]] = []


@lru_cache(maxsize=1)
def load_modane_raw(path: str = str(MODANE_PATH)) -> np.ndarray:
    """Full record as a read-only 1-D float64 array (cached in-process)."""
    from scipy.io import loadmat
    v = loadmat(path)['Vit'][:, 0]
    v.setflags(write=False)
    return v


def select_windows(n_windows, n=None, spread=False, seed=0, start_window=0):
    """Indices of the windows `load_modane_windows` keeps (see its docstring)."""
    if spread:
        if n is None or n > n_windows:
            raise ValueError(f'spread=True needs n <= number of windows ({n_windows}), got {n}')
        return np.sort(np.random.default_rng(seed).choice(n_windows, n, replace=False))
    return np.arange(n_windows)[start_window:None if n is None else start_window + n]


def load_modane_windows(window_len=1024, target_len=256, n=None, start_window=0,
                        spread=False, seed=0, device='cpu', normalized=True):
    """Windows exactly as `run_jet_SDE.py` builds them.

    Record -> non-overlapping windows of `window_len` (each detrended to be
    periodic by `split_periodize_reshape`) -> Db3 low-pass + decimation until
    the length is `target_len` -> selection -> global standardization
    (`normalize`, over the selected windows).

    Selection: windows [start_window, start_window + n) (`spread=False`), or
    `n` windows drawn at random without replacement from the whole record with
    a fixed `seed`, kept in record order (`spread=True`). Consecutive windows
    are strongly correlated (integral scale ~2100 samples, rho(1024) ~ 0.5), so
    spreading them gives many more independent samples for the same n.

    Returns
    -------
    x : torch.Tensor, shape (n, 1, target_len), float32
    starts : np.ndarray, shape (n,), sample index in the raw record where each
        window begins.
    """
    from codes.utils import split_periodize_reshape, normalize
    from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet

    if EXCLUDED_INTERVALS:
        raise NotImplementedError('window selection around excluded intervals')

    raw = torch.from_numpy(load_modane_raw().astype(np.float32)[None, None, :])
    x = split_periodize_reshape(raw, window_len)
    starts = np.arange(x.shape[0]) * window_len

    n_scales = int(np.log2(window_len / target_len)) if window_len > target_len else 0
    if n_scales:
        W = DefineWavelet('Db', m=3, device=device)
        x = x.to(device)
        for _ in range(n_scales):
            x = W.decompose(x)[1]

    idx = select_windows(x.shape[0], n, spread, seed, start_window)
    x, starts = x[torch.as_tensor(idx, device=x.device)], starts[idx]
    if normalized:
        x = normalize(x)
    return x.to(device), starts


def describe_modane(window_len=1024, n1=5000):
    """Print what the record is: size, mean flow, fluctuation level, and how
    many windows (and roughly independent ones) it gives."""
    v = load_modane_raw()
    N, U, s = v.size, v.mean(), v.std()
    n_win = N // window_len
    print(f'Modane record: N = {N:,} samples   {MODANE_META}')
    print(f'  U = {U:.3f}   sigma = {s:.3f}   sigma/U = {s / U:.3f}   '
          f'min/max = {v.min():.2f}/{v.max():.2f}')
    print(f'  windows of {window_len}: {n_win:,}  (n1 = {n1:,} -> {n1 / n_win:.0%} of the record)')
    print(f'  excluded intervals: {EXCLUDED_INTERVALS or "none"}')
