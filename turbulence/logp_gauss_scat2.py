#!/usr/bin/env python3
"""log p of the Gaussian scattering-2 baseline: width, cross-seed agreement, existence.

Question: does the Gaussian maxent model (turbulence_final_gauss_scat2, only
Scattering_Second_Order_Q1) concentrate log p, compared with the full turbulence_final
model? For a well-fitted Gaussian in M = 256 dimensions, log p on the model's own
samples has std sqrt(M/2) = sqrt(128) ~ 11.3 nats (chi^2 with M degrees of freedom).

log p(x) = theta . phi(x) - log Z, theta = Theta_reg(t=1). log Z is not computed: every
log p is shown relative to its own seed's model-sample mean,
    dlogp(x) = theta . phi(x) - < theta . phi >_model samples,
so log Z cancels exactly and seeds are comparable. On data, dlogp says how much less
(or more) likely a data window is than a typical model sample.

Per seed: the SDE's final walkers, Theta_reg (resolve_theta_reg.py output), phi via the
same _phi helper as the entropy pipeline. The data x1 is rebuilt exactly as run_SDE does
(split, Db3 coarse-graining to target_len, first n1, normalise) and shared by all seeds.

Checks printed to stdout:
  - existence of the Gaussian: theta_j < 0 for every j (what was asked; sufficient), and
    the exact condition s(w) = sum_j theta_j |psi_j(w)|^2 < 0 at every frequency
    (+-w paired, x is real). With overlapping filters a positive theta_j can still give
    a normalisable Gaussian, and the exact test tells the two cases apart. Per seed and
    for the across-seed mean theta.
  - width: std of dlogp on model (vs sqrt(M/2)) and on data, per seed
  - cross-seed agreement: Pearson of dlogp on data (same x1, pointwise) and KS distance
    of the model dlogp distributions (different samples, shape only)

Outputs in --outdir: <cfg>_logp.npz per seed, all_seeds.npz, summary.csv,
logp_hist.png (model and data panels, one curve per seed, N(0, M/2) reference),
logp_scatter_data.png (pairwise seed-vs-seed dlogp on data, Pearson in each cell).

Usage (JZ, repo root; TURBULENCE_1D_DATA_PATH optional, data_loader.py has the default):
  python turbulence/logp_gauss_scat2.py \
      --group_dir turbulence/experiments/turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_terms9f3f9a22_turbulence_final_gauss_scat2 \
      --outdir turbulence/saved_results/gauss_scat2_logp
"""
import argparse, csv, json, sys
from pathlib import Path
from itertools import combinations
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'turbulence'))
from utils import normalize                                                     # noqa: E402
from utils_entropy import _phi                                                  # noqa: E402
from filters_bank import return_Filters                                         # noqa: E402
from potentials_builder import get_1d_potentials                                # noqa: E402
# same helpers run_SDE uses to build x1 (it star-imports them from codes/ and data/)
from run_SDE import split_periodize_reshape, load_turbulence_1d, DefineWavelet  # noqa: E402
from gauss_spectrum import sample_gaussian_spectrum, true_log_p                 # noqa: E402


# ---------------------------------------------------------------- loading

def build_pots(cfg, device):
    J, Q, M = cfg['J'], cfg['Q'], cfg['M']
    wav = cfg.get('wav_type', 'morlet')
    filters, filters_Phi = return_Filters(M, J, 1, device=device, include_phi=True, wav_type=wav)
    filters_Q = return_Filters(M, J, Q, device=device, wav_type=wav)
    return get_1d_potentials(cfg['terms'], J, filters, Q,
                             filters_Q=filters_Q, filters_Phi=filters_Phi,
                             deduplicate_filters=bool(cfg.get('deduplicate_filters') or False),
                             dedup_tol=cfg.get('dedup_tol') or 1e-12)


def load_gauss_spectrum(exp_dir):
    """S (C, M) float64 saved by run_SDE / run_jet_SDE --gauss_spectrum, or None for a real-data run."""
    f = exp_dir / 'gauss_spectrum.pt'
    if not f.exists():
        return None, None
    blob = torch.load(f, map_location='cpu', weights_only=False)
    return blob['S'], int(blob.get('data_seed', 0))


def load_data_x1(cfg, device, exp_dir=None):
    """x1 exactly as run_SDE builds it (non-spread branch): load_turbulence_1d,
    split_periodize_reshape to subseries_len, Db3 coarse-graining down to target_len,
    first n1, normalise.
    --gauss_spectrum runs (turbulence or jets): x1 is redrawn from the saved spectrum S with the
    same generator and seed, so it is bit-identical to the run's x1 (codes/gauss_spectrum.py)."""
    if cfg.get('gauss_spectrum'):
        S, data_seed = load_gauss_spectrum(exp_dir)
        if S is None:
            raise FileNotFoundError(f'gauss_spectrum run without {exp_dir}/gauss_spectrum.pt')
        return sample_gaussian_spectrum(S, int(cfg['n1']), seed=data_seed).float().to(device)
    if cfg.get('n1_spread'):
        raise NotImplementedError('n1_spread runs: data selection not reproduced here')
    raw = load_turbulence_1d().to(device)
    data = split_periodize_reshape(raw, cfg['subseries_len'])                   # (B, 1, pre_len)
    pre_len = data.shape[-1]
    if pre_len > cfg['target_len']:
        W = DefineWavelet('Db', m=3, device=device)
        for _ in range(int(np.log2(pre_len / cfg['target_len']))):
            data = W.decompose(data)[1]
    x1 = normalize(data[:cfg['n1']]).to(device)
    if x1.shape[-1] != cfg['M']:
        raise RuntimeError(f'data length {x1.shape[-1]} != model M {cfg["M"]}')
    return x1


def load_theta_reg(sweep_root, config_name, lam=None, ridge=None):
    """resolve_theta_reg.py writes <sweep_root>/<config>/lam<L>_ridge<R>{suffix}.pt.
    Exactly one match is required; --lam / --ridge pick one if there are several."""
    cfg_dir = sweep_root / config_name
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f'no Theta_reg dir {cfg_dir}')
    files = sorted(cfg_dir.glob('lam*_ridge*.pt'))
    if lam is not None:
        files = [p for p in files if p.name.startswith(f'lam{lam:g}_')]
    if ridge is not None:
        files = [p for p in files if f'_ridge{ridge:g}' in p.name]
    if len(files) != 1:
        raise RuntimeError(f'{len(files)} Theta_reg files match under {cfg_dir} '
                           f'(pin --lam/--ridge): {[p.name for p in files]}')
    blob = torch.load(files[0], map_location='cpu', weights_only=False)
    return blob['Theta_reg'], files[0].name


def load_samples(results_root, name):
    # not utils.load_results: it torch.load()s saved_results/lagrange_multipliers_regularised/
    # <cfg> whenever that path exists, and resolve_theta_reg.py makes it a directory
    path = results_root / 'saved_results' / 'samples' / f'{name}.pt'
    if not path.exists():
        path = path.with_name(name)                                             # pre-.pt convention
    return torch.load(path, map_location='cpu', weights_only=False)


def theta_phi(x, theta_1, pots, chunk, device):
    """theta . phi(x) per sample, chunked, float64 numpy."""
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            phi = _phi(x[i:i + chunk].to(device), pots).double().cpu()          # (b, r)
            out.append((phi @ theta_1).numpy())
    return np.concatenate(out)


def one_seed(exp_dir, args, x_data, sweep_root):
    cfg = json.loads((exp_dir / 'config.json').read_text())
    cfg = cfg.get('args', cfg)
    name, seed = exp_dir.name, int(cfg['seed'])
    Theta_reg, fname = load_theta_reg(sweep_root, name, args.lam, args.ridge)
    xt = load_samples(args.results_root, name)
    pots = build_pots(cfg, args.device)
    theta_1 = Theta_reg[-1].detach().cpu().double()                             # (r,)
    tp_model = theta_phi(xt, theta_1, pots, args.chunk, args.device)
    tp_data = theta_phi(x_data, theta_1, pots, args.chunk, args.device)
    shift = tp_model.mean()                                                     # removes log Z
    r = dict(seed=seed, theta=theta_1.numpy(), shift=shift,
             logp_model=tp_model - shift, logp_data=tp_data - shift,
             filters=filters_of(pots))
    extra = {}
    S, _ = load_gauss_spectrum(exp_dir)
    if S is not None:
        # exact log p of the true Gaussian, relative to its mean over the true law's own samples
        # (x1), as the MGD log p is relative to its mean over the model's own samples
        ex_data = true_log_p(x_data.cpu(), S).numpy()
        ex_model = true_log_p(xt.reshape(xt.shape[0], 1, -1).cpu(), S).numpy()
        ex_shift = ex_data.mean()
        r['exact_data'], r['exact_model'] = ex_data - ex_shift, ex_model - ex_shift
        extra = dict(exact_data=r['exact_data'], exact_model=r['exact_model'])
    np.savez(args.outdir / f'{name}_logp.npz', seed=seed, theta=r['theta'],
             logp_model=r['logp_model'], logp_data=r['logp_data'],
             theta_phi_model_mean=shift, theta_reg_file=fname, **extra)
    print(f'  seed {seed}: {fname} | model std {r["logp_model"].std():7.3f} | '
          f'data std {r["logp_data"].std():7.3f}, mean {r["logp_data"].mean():+8.2f}, '
          f'min {r["logp_data"].min():+9.2f}')
    if extra:
        print(f'           exact log p: on data std {r["exact_data"].std():7.3f} | on model samples std '
              f'{r["exact_model"].std():7.3f}, mean {r["exact_model"].mean():+8.2f} (0 = as typical as the data)'
              f' | Pearson(MGD, exact) on data {pearsonr(r["logp_data"], r["exact_data"])[0]:.4f}')
    return r


def filters_of(pots):
    """|psi_j(w)|^2 of the second-order scattering term, (r, M) numpy -- or None when the
    model has other terms (then the exact existence test below does not apply)."""
    if list(pots) != ['Scattering_Second_Order_Q1']:
        return None
    f = pots['Scattering_Second_Order_Q1'].filters
    return (f.reshape(-1, f.shape[-1]).abs() ** 2).double().cpu().numpy()


# ---------------------------------------------------------------- checks

def existence(theta, psi2):
    """(n theta_j >= 0, max over frequency pairs of s(w) + s(-w)). Gaussian exists iff the
    second number is < 0: theta . phi(x) = (1/M^2) sum_w s(w) |x^(w)|^2 with
    s(w) = sum_j theta_j |psi_j(w)|^2, and |x^(w)| = |x^(-w)| for real x."""
    n_pos = int((theta >= 0).sum())
    if psi2 is None:
        return n_pos, float('nan')
    s = theta @ psi2                                                            # (M,)
    M = s.size
    pair = s + s[(-np.arange(M)) % M]                                           # s(w) + s(-w)
    return n_pos, float(pair[:M // 2 + 1].max())


def check_existence(res):
    print('\nexistence of the Gaussian (needs theta_j < 0 for all j: sufficient; '
          'max_w s(w)+s(-w) < 0: exact):')
    rows = [(f'seed {r["seed"]}', r['theta']) for r in res]
    rows.append(('across-seed mean', np.mean([r['theta'] for r in res], axis=0)))
    psi2 = res[0]['filters']
    for label, th in rows:
        n_pos, smax = existence(th, psi2)
        ok = n_pos == 0 and not smax >= 0
        verdict = 'OK' if ok else ('exists (exact test) but some theta_j >= 0'
                                   if smax < 0 else 'NOT NORMALISABLE')
        print(f'  {label:>17}: theta in [{th.min():+.3e}, {th.max():+.3e}], '
              f'{n_pos}/{th.size} >= 0, max s = {smax:+.3e}  -> {verdict}')


def pairwise(arrays, fn):
    n = len(arrays); out = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        out[i, j] = out[j, i] = fn(arrays[i], arrays[j])
    return out


def print_matrix(label, seeds, mat):
    print(f'\n{label}:')
    print('  seed ' + ''.join(f'{s:>8}' for s in seeds))
    for i, s in enumerate(seeds):
        print(f'  {s:>4} ' + ''.join('       -' if i == j else f'{mat[i, j]:8.3f}'
                                     for j in range(len(seeds))))


# ---------------------------------------------------------------- figures

def plot_hists(res, out_png, M):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(res)))
    for ax, key, title in zip(axes, ('logp_model', 'logp_data'), ('model samples', 'data')):
        vals = np.concatenate([r[key] for r in res])
        bins = np.linspace(*np.quantile(vals, [0.001, 0.999]), 80)
        for r, c in zip(res, colors):
            ax.hist(r[key], bins=bins, histtype='step', density=True, color=c, lw=1.3,
                    label=f'seed {r["seed"]}, std {r[key].std():.2f}')
        sd = np.sqrt(M / 2)
        x = np.linspace(bins[0], bins[-1], 400)
        ax.plot(x, np.exp(-x ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), 'k--', lw=1,
                label=f'N(0, M/2), std {sd:.2f}')
        ax.set_xlabel(r'log p(x) $-$ $\langle$log p$\rangle_{\rm model}$ (nats)')
        ax.set_ylabel('density'); ax.set_title(title); ax.legend(fontsize=7)
    fig.suptitle('Gaussian scattering-2 model: log p relative to the typical model sample')
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_scatter_data(res, out_png, n_show=2000):
    """Seed i vs seed j, dlogp on the same data windows (lower triangle), per-seed
    histogram on the diagonal. Points on y = x: the two seeds give that window the same
    log p. Pearson on all windows; only n_show random windows drawn."""
    arrs = [r['logp_data'] for r in res]; n = len(arrs)
    idx = np.random.default_rng(0).choice(arrs[0].size, min(n_show, arrs[0].size), replace=False)
    lo = min(a.min() for a in arrs); hi = max(a.max() for a in arrs)
    pad = 0.03 * (hi - lo); lim = (lo - pad, hi + pad)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(arrs[i], bins=50, color='steelblue'); ax.set_xlim(lim); ax.set_yticks([])
            elif i > j:
                ax.plot(lim, lim, color='0.6', lw=0.7)
                ax.scatter(arrs[j][idx], arrs[i][idx], s=3, alpha=0.3)
                ax.text(0.04, 0.94, f'r={pearsonr(arrs[j], arrs[i])[0]:.3f}',
                        transform=ax.transAxes, va='top', fontsize=8)
                ax.set_xlim(lim); ax.set_ylim(lim)
            else:
                ax.axis('off')
            if i == n - 1: ax.set_xlabel(f'seed {res[j]["seed"]}')
            if j == 0 and i > 0: ax.set_ylabel(f'seed {res[i]["seed"]}')
    fig.suptitle(r'log p $-$ $\langle$log p$\rangle_{\rm model}$ on data: seed vs seed')
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group_dir', type=Path, required=True,
                   help='experiments/<group>/, the parent of the per-seed folders')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=ROOT / 'turbulence',
                   help='folder holding saved_results/ (default: turbulence/)')
    p.add_argument('--theta_reg_dir', type=Path, default=None,
                   help='resolve_theta_reg.py --outdir (default: <results_root>/saved_results/'
                        'lagrange_multipliers_regularised)')
    p.add_argument('--lam', type=float, default=None, help='pick lam if several files')
    p.add_argument('--ridge', type=float, default=None, help='pick ridge if several files')
    p.add_argument('--chunk', type=int, default=1000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    sweep_root = args.theta_reg_dir or args.results_root / 'saved_results' / 'lagrange_multipliers_regularised'

    seed_dirs = sorted(d for d in args.group_dir.iterdir() if (d / 'config.json').exists())
    print(f'{len(seed_dirs)} seed folders with config.json under {args.group_dir.name}')
    if not seed_dirs:
        return
    cfg0 = json.loads((seed_dirs[0] / 'config.json').read_text()); cfg0 = cfg0.get('args', cfg0)
    x_data = load_data_x1(cfg0, args.device, seed_dirs[0])
    M = int(cfg0['M'])
    print(f'data x1 {tuple(x_data.shape)}; theory: std(log p) on model samples = '
          f'sqrt(M/2) = {np.sqrt(M / 2):.3f}')

    res = []
    for d in seed_dirs:
        try:
            res.append(one_seed(d, args, x_data, sweep_root))
        except Exception as e:
            print(f'  {d.name}: FAIL {type(e).__name__}: {e}')
    if not res:
        return
    seeds = [r['seed'] for r in res]

    check_existence(res)

    print('\nwidth of log p (std, nats):')
    for r in res:
        print(f'  seed {r["seed"]}: model {r["logp_model"].std():7.3f} '
              f'(theory {np.sqrt(M / 2):.3f}) | data {r["logp_data"].std():7.3f}')

    if len(res) > 1:
        R = pairwise([r['logp_data'] for r in res], lambda a, b: pearsonr(a, b)[0])
        K = pairwise([r['logp_model'] for r in res], lambda a, b: ks_2samp(a, b).statistic)
        print_matrix('cross-seed Pearson of log p on data (same windows)', seeds, R)
        print_matrix('cross-seed KS distance of log p on model samples (shape)', seeds, K)
        iu = np.triu_indices(len(res), 1)
        print(f'\nPearson on data in [{R[iu].min():.3f}, {R[iu].max():.3f}]; '
              f'KS on model in [{K[iu].min():.3f}, {K[iu].max():.3f}]')

    exact = {k: np.stack([r[k] for r in res]) for k in ('exact_data', 'exact_model')
             if all(k in r for r in res)}
    np.savez(args.outdir / 'all_seeds.npz', seeds=np.array(seeds), M=M,
             theta=np.stack([r['theta'] for r in res]),
             logp_model=np.stack([r['logp_model'] for r in res]),
             logp_data=np.stack([r['logp_data'] for r in res]), **exact)
    with open(args.outdir / 'summary.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'kind', 'n', 'mean', 'std', 'min', 'max', 'std_minus_sqrt_M_over_2'])
        for r in res:
            for kind in ('model', 'data'):
                a = r[f'logp_{kind}']
                w.writerow([r['seed'], kind, a.size, a.mean(), a.std(), a.min(), a.max(),
                            a.std() - np.sqrt(M / 2)])
    plot_hists(res, args.outdir / 'logp_hist.png', M)
    plot_scatter_data(res, args.outdir / 'logp_scatter_data.png')
    print(f'\nwrote {args.outdir}: logp_hist.png, logp_scatter_data.png, all_seeds.npz, '
          f'summary.csv, {len(res)} per-seed .npz')


if __name__ == '__main__':
    main()
