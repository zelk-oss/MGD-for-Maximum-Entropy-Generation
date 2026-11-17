#!/usr/bin/env python3
"""(a) of the whitening test (2026-11-13): spectral range of each data set as MGD sees it.

For each data set, x1 is built as in its entry point (windows, coarse-graining, normalize), then
    C(w) = mean_n |FFT_ortho(x_n)(w)|^2
= the eigenvalues of the stationary (circulant) part of the data covariance, in the units of the MGD
reference noise x0 ~ N(0, I), whose ortho-FFT modes have variance 1.

Printed per data set (C sorted; w = 0 included and also excluded, because normalize removes only the global mean):
  kappa      = C_max / C_min                  condition number of the data covariance (circulant part)
  kappa_1pct = C_max / (1 % quantile of C)    the same without the 1 % weakest modes
  1-t*       = (2/pi) arctan(sqrt(C_min))     time left when the weakest mode stops being noise-dominated in the Cos
               interpolant x_t = cos(pi t/2) x0 + sin(pi t/2) x1 (per-mode variance cos^2 + sin^2 C: equal terms at
               tan(pi t/2) = 1/sqrt(C))
  2 C_min    = largest sigma^2 h for which an explicit Euler step of the Langevin drift sigma^2 grad log p is stable on
               the Gaussian with this covariance (x <- (1 - sigma^2 h / C) x + noise; stable iff sigma^2 h < 2 C)
Outputs: <out>/kappa_spectra.csv, <out>/spectra.npz (sorted C per data set), <out>/kappa_spectra.png.

Usage (JZ, repo root): python whitening_test/kappa_spectra.py --out whitening_test/kappa [--n1 10000]
"""
import argparse, csv, sys, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'jets'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def turb1d(n1, sublen, M):
    from data.data_loader import load_turbulence_1d
    from codes.utils import split_periodize_reshape, normalize
    from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet
    data = split_periodize_reshape(load_turbulence_1d(), sublen)
    W = DefineWavelet('Db', m=3, device='cpu')
    for _ in range(int(np.log2(data.shape[-1] / M))):
        data = W.decompose(data)[1]
    return normalize(data[:n1])


def jets(n1, sublen, M):
    from jet_data import load_modane_windows
    x, _ = load_modane_windows(sublen, M, n=n1, spread=False)
    return x


def turb2d(n1, fact=2):
    from codes.utils import normalize
    sys.path.insert(0, str(ROOT / 'turbulence_2d'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('t2d_run', ROOT / 'turbulence_2d' / 'run_SDE.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    path = ROOT / 'data' / 'data_files' / 'ns_randn4_N256_c1.mat'
    return normalize(mod.load_turbulence_2d(path, fact=fact, n_max=n1))


def spectrum(x):
    """mean_n |FFT_ortho(x_n)|^2 over all spatial axes, flattened, float64."""
    x = x.double()
    dims = tuple(range(2, x.dim()))
    xf = torch.fft.fftn(x, dim=dims, norm='ortho')
    return (xf.abs() ** 2).mean((0, 1)).reshape(-1).numpy()


def summary(name, C):
    Cs = np.sort(C)
    nz = np.ones(C.size, bool); nz[0] = False          # w = 0 is index 0 of the flattened FFT grid
    Cnz = np.sort(C[nz])
    row = dict(dataset=name, n_modes=C.size, mean_C=C.mean(), C_max=Cs[-1],
               C_min=Cs[0], C_min_no0=Cnz[0], C_q1pct=np.quantile(Cnz, 0.01), C_q10pct=np.quantile(Cnz, 0.10),
               C_median=np.median(Cnz))
    row['kappa'] = row['C_max'] / row['C_min']
    row['kappa_no0'] = row['C_max'] / row['C_min_no0']
    row['kappa_1pct'] = row['C_max'] / row['C_q1pct']
    row['one_minus_t_star'] = 2 / np.pi * np.arctan(np.sqrt(row['C_min_no0']))
    row['two_C_min'] = 2 * row['C_min_no0']
    return row, Cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n1', type=int, default=10000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [('turb1d_M256', lambda: turb1d(args.n1, 512, 256)),
            ('turb1d_M64', lambda: turb1d(args.n1, 512, 64)),
            ('turb1d_M32', lambda: turb1d(args.n1, 512, 32)),
            ('jets_modane_M256', lambda: jets(args.n1, 512, 256)),
            ('turb2d_128x128', lambda: turb2d(1000))]
    rows, spectra = [], {}
    for name, fn in jobs:
        try:
            x = fn()
            row, Cs = summary(name, spectrum(x))
            row['n_signals'] = x.shape[0]
            rows.append(row); spectra[name] = Cs
            print(f"{name:18s} n {x.shape[0]:6d} modes {row['n_modes']:6d} | mean C {row['mean_C']:.3f} | "
                  f"C_max {row['C_max']:.3e}  C_min(no w=0) {row['C_min_no0']:.3e}  q1% {row['C_q1pct']:.3e} | "
                  f"kappa {row['kappa_no0']:.2e}  kappa_1% {row['kappa_1pct']:.2e} | 1-t* {row['one_minus_t_star']:.2e} | "
                  f"2 C_min {row['two_C_min']:.2e}", flush=True)
        except Exception:
            print(f'{name}: FAILED'); traceback.print_exc()
    if not rows:
        sys.exit(1)
    keys = ['dataset', 'n_signals', 'n_modes', 'mean_C', 'C_max', 'C_min', 'C_min_no0', 'C_q1pct', 'C_q10pct', 'C_median',
            'kappa', 'kappa_no0', 'kappa_1pct', 'one_minus_t_star', 'two_C_min']
    with open(args.out / 'kappa_spectra.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    np.savez(args.out / 'spectra.npz', **spectra)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        Cs = spectra[r['dataset']]
        ax.plot(np.linspace(0, 1, Cs.size), Cs[::-1], label=f"{r['dataset']}  (kappa {r['kappa_no0']:.1e}, w=0 excluded)")
    ax.axhline(1.0, color='k', ls=':', lw=1, label='variance of a white-noise mode (x0 ~ N(0, I))')
    ax.set_yscale('log'); ax.set_xlabel('modes, sorted by decreasing variance (fraction of all modes)')
    ax.set_ylabel('C(w) = mean |FFT_ortho x|^2  (normalized data)')
    ax.set_title('Spectral range of the data: modes far below 1 appear only near t = 1 and set the Euler step')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.out / 'kappa_spectra.png', dpi=120)
    print(f"wrote {args.out / 'kappa_spectra.csv'}, spectra.npz, kappa_spectra.png")


if __name__ == '__main__':
    main()
