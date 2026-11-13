"""Calibration of a sample-based sigma-convergence criterion on the Gaussian M-scan runs (2026-11-12).

Question: when the exact deficit A = H(p*) - H(gen) of the MGD output is known (Gaussian target), at which sigma does a
two-sample distance between the generated samples at sigma and sigma' fall to its sampling-noise level, and does that
agree with where A itself saturates?  If yes, the distance can be used on real data (where A is unknown).

Distances between two sets X (n, M), Y (n, M), subsampled to n_sub each, repeated R times:
  energy   unbiased energy distance  D_E = 2 E|X-Y| - E|X-X'| - E|Y-Y'|   (>= 0 in expectation, 0 iff equal laws)
  sw2      sliced Wasserstein-2 squared (n_dir random unit directions; biased upward, floor from the null)
  klspec   symmetrised KL between the circulant Gaussians with the mean spectra S(w) = mean |FFT x|^2 of X and Y
           (the most sensitive test available to a Gaussian family: it uses stationarity; a reference ceiling)
Null (noise floor): the same metric between two disjoint halves of ONE sample (same n_sub). Test pairs:
  sigma pairs at the reference nt, and nt controls (same sigma, two nt) that should sit at the noise floor.

Usage: python gauss_spectrum/sample_distance_calibration.py --root gauss_spectrum/m_scan --Ms 32 64 --outdir <dir>
       python gauss_spectrum/sample_distance_calibration.py --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_samples(root, sigma, nt):
    path_dir = Path(root) / 'saved_results' / 'samples'
    files = sorted(path_dir.glob(f'*_ntsweep_s{sigma:g}_nt{nt}_*.pt'))
    if len(files) != 1:
        raise RuntimeError(f'{len(files)} sample files for sigma {sigma:g} nt {nt} in {path_dir}')
    x = torch.load(files[0], map_location='cpu', weights_only=False)
    return x.reshape(x.shape[0], -1).double()


def energy_distance(X, Y):
    n, m = len(X), len(Y)
    dxy = torch.cdist(X, Y).mean()
    dxx = torch.cdist(X, X).sum() / (n * (n - 1))
    dyy = torch.cdist(Y, Y).sum() / (m * (m - 1))
    return float(2 * dxy - dxx - dyy)


def sliced_w2(X, Y, gen, n_dir=500):
    U = torch.randn(X.shape[1], n_dir, generator=gen, dtype=X.dtype)
    U = (U / U.norm(dim=0)).to(X.device)
    px, _ = torch.sort(X @ U, dim=0)
    py, _ = torch.sort(Y @ U, dim=0)
    return float(((px - py) ** 2).mean())


def mean_spectrum(X):
    xh = torch.fft.fft(X, dim=-1)
    return (xh.real ** 2 + xh.imag ** 2).mean(0)


def kl_spec(X, Y):
    """Symmetrised KL (nats) between the circulant Gaussians with the mean spectra of X and Y."""
    r = mean_spectrum(X) / mean_spectrum(Y)
    kl_xy = 0.5 * float((r - 1 - torch.log(r)).sum())
    kl_yx = 0.5 * float((1 / r - 1 + torch.log(r)).sum())
    return 0.5 * (kl_xy + kl_yx)


def metrics(X, Y, gen):
    return dict(energy=energy_distance(X, Y), sw2=sliced_w2(X, Y, gen), klspec=kl_spec(X, Y))


def summarize(rows):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}


def run(sets, tests, nulls, n_sub, reps, device, seed=0):
    """sets: {key: (n, M) tensor}; tests: [(label, keyA, keyB)]; nulls: [key]. Returns {label: {metric: (mean, std)}}."""
    gen = torch.Generator().manual_seed(seed)
    out = {}

    def sub(X, idx):
        return X[idx].to(device)

    for key in nulls:
        X = sets[key]
        rows = []
        for _ in range(reps):
            p = torch.randperm(len(X), generator=gen)
            rows.append(metrics(sub(X, p[:n_sub]), sub(X, p[n_sub:2 * n_sub]), gen))
        out[f'null {key}'] = summarize(rows)
    for label, ka, kb in tests:
        A, B = sets[ka], sets[kb]
        rows = []
        for _ in range(reps):
            pa = torch.randperm(len(A), generator=gen)[:n_sub]
            pb = torch.randperm(len(B), generator=gen)[:n_sub]
            rows.append(metrics(sub(A, pa), sub(B, pb), gen))
        out[label] = summarize(rows)
    return out


def table(res):
    lines = [f'{"comparison":42s} {"energy":>22s} {"sw2":>22s} {"klspec [nats]":>22s}']
    for k, v in res.items():
        f = lambda m: f'{v[m][0]:+.3e} +- {v[m][1]:.1e}'
        lines.append(f'{k:42s} {f("energy"):>22s} {f("sw2"):>22s} {f("klspec"):>22s}')
    return '\n'.join(lines)


def selftest():
    torch.manual_seed(0)
    M, n = 16, 6000
    k = torch.arange(M)
    S0 = 1.0 / (1.0 + torch.minimum(k, M - k).double()) ** 2 + 1e-3

    def draw(S, seed):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n, M, generator=g, dtype=torch.float64)
        return torch.fft.ifft(torch.fft.fft(w, dim=-1) * torch.sqrt(S / M), dim=-1).real

    sets = {'a': draw(S0, 1), 'a2': draw(S0, 2), 'b': draw(S0 * 1.05, 3), 'c': draw(S0 * 1.5, 4)}
    res = run(sets, [('same law (a vs a2)', 'a', 'a2'), ('spectrum x1.05 (a vs b)', 'a', 'b'),
                     ('spectrum x1.50 (a vs c)', 'a', 'c')], ['a'], n_sub=2500, reps=5, device='cpu')
    print(table(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('gauss_spectrum/m_scan'))
    p.add_argument('--Ms', type=int, nargs='+', default=[32, 64])
    p.add_argument('--sigmas', type=float, nargs='+', default=[1.0, 2.0, 5.0])
    p.add_argument('--nt_ref', type=int, default=1600000)
    p.add_argument('--nt_ctrl', type=int, default=400000)
    p.add_argument('--n_sub', type=int, default=5000)
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--outdir', type=Path, default=Path('gauss_spectrum/m_scan/sample_distance'))
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.outdir.mkdir(parents=True, exist_ok=True)
    for M in args.Ms:
        root = args.root / f'M{M}' / 'turbulence'
        sets = {}
        for s in args.sigmas:
            sets[(s, args.nt_ref)] = load_samples(root, s, args.nt_ref)
            sets[(s, args.nt_ctrl)] = load_samples(root, s, args.nt_ctrl)
        tests = []
        for i, sa in enumerate(args.sigmas):
            for sb in args.sigmas[i + 1:]:
                tests.append((f'sigma {sa:g} vs {sb:g} (nt {args.nt_ref})', (sa, args.nt_ref), (sb, args.nt_ref)))
        for s in args.sigmas:
            tests.append((f'control sigma {s:g}: nt {args.nt_ctrl} vs {args.nt_ref}', (s, args.nt_ctrl), (s, args.nt_ref)))
        nulls = [(s, args.nt_ref) for s in args.sigmas]
        res = run(sets, tests, nulls, args.n_sub, args.reps, device)
        res = {k.replace("null (", "null sigma ").replace(f", {args.nt_ref})", f" (nt {args.nt_ref}, two halves)"): v
               for k, v in res.items()}
        print(f'\n===== M = {M}  (n_sub {args.n_sub}, {args.reps} repeats, device {device}); mean +- std over repeats =====')
        print(table(res))
        (args.outdir / f'sample_distance_M{M}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
