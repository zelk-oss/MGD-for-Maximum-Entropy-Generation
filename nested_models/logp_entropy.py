"""log p, entropy and concentration of one finished run (the part the statistics notebooks lack).

Convention of the codebase: p_theta(x) = exp(+theta . phi(x)) / Z, so
    -log p(x) = log Z - theta . phi(x).
theta = Theta_reg[-1] solved offline (resolve_lamsweep.sh, lam 5e-4, ridge 1e-4, unconstrained).
log Z is not known exactly: as in compare_entropy_methods.py, log_Z_lb = H_bound + theta . m_target,
with H_bound the MGD entropy lower bound (Prop. 4.3) and m_target the data moments at t = 1.
Then, with x ~ the run's final samples,
    mean(-log p) = H_sample = log_Z_lb - theta . <phi>_samples = H_bound + theta . (m_target - <phi>_samples)
so H_sample and H_bound differ exactly by the moment-matching residual projected on theta. The WIDTH
of -log p (std) does not depend on log Z, so concentration (std, std / H) is exact up to that residual.
The decomposition C_F(x) = theta_F . phi_F(x) gives the contribution of each potential family.

Everything reuses the repo (phi_matrix, theta_column_map, entropy_bound, build_data); the only new
loader code is the rebuild of the fitted potentials, which also handles the decimated-DWT arm
(--dwt_order) that logp_family_contrib / compare_entropy_methods cannot.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

_BACKEND = matplotlib.get_backend()        # codes/potentials/potentials_1d.py forces 'Agg' at import: restored below

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TURB = ROOT / 'turbulence'
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', TURB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters_bank import return_Filters                                   # noqa: E402
from potentials_builder import get_1d_potentials                           # noqa: E402
from check_fit_offline import build_data                                   # noqa: E402
from codes.gram_preflight import stat_labels                               # noqa: E402
from codes.utils_entropy import entropy_bound                              # noqa: E402

try:                                        # a notebook keeps its inline backend, a script keeps its own
    matplotlib.use(_BACKEND)
except Exception:
    pass

CHUNK = 500


def phi_matrix(pots, x):
    """(n, r) float64 numpy: every potential's statistics on x, chunked, in the potentials-dict column order.
    (Copied from turbulence/logp_family_contrib.py, which cannot be imported here: it calls
    matplotlib.use('Agg') at import and would switch a notebook off the inline backend.)"""
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], CHUNK):
            feats = []
            for p in pots.values():
                f = p(x[i:i + CHUNK])
                feats.append(f.unsqueeze(1) if f.ndim == 1 else f)
            out.append(torch.cat(feats, dim=-1).detach().cpu().double())
    return torch.cat(out).numpy()


RUN_NAME_RE = re.compile(r'_seed_(\d+)_terms([0-9a-f]{8})_(.+)_(\d{8}_\d{4})$')       # same as run_comparison


def find_run(label, seed, root=TURB):
    """(exp_dir, config) of the newest run with this label and seed that has saved samples
    (so the pre-flight-only runs of the same label are skipped)."""
    cands = sorted((root / 'experiments').glob(f'*/*_seed_{seed}_*{label}_2*'), key=lambda f: f.stat().st_mtime)
    cands = [c for c in cands if (m := RUN_NAME_RE.search(c.name)) and m[3] == label      # exact label, as run_comparison
             and (root / 'saved_results' / 'samples' / f'{c.name}.pt').exists()]
    if not cands:
        raise FileNotFoundError(f'no finished run with samples for label {label!r}, seed {seed} under {root}')
    return cands[-1], cands[-1].name


def load_theta(config, lam=5e-4, ridge=1e-4, root=TURB):
    """Theta_reg[-1] (float64, code sign) of the UNCONSTRAINED offline solve, and the file name."""
    for sub in ('theta_reg_lamsweep', 'lagrange_multipliers_regularised'):
        d = root / 'saved_results' / sub / config
        if d.is_dir():
            fs = [f for f in sorted(d.glob(f'lam{lam:g}_ridge{ridge:g}*.pt')) if 'neg' not in f.name]
            if len(fs) == 1:
                blob = torch.load(fs[0], map_location='cpu', weights_only=False)
                return blob['Theta_reg'][-1].double().numpy(), f'{sub}/{fs[0].name}'
            if len(fs) > 1:
                raise RuntimeError(f'{len(fs)} Theta_reg files for {config} in {d}: {[f.name for f in fs]}')
    raise FileNotFoundError(f'no unconstrained Theta_reg (lam {lam:g}, ridge {ridge:g}) for {config}')


def rebuild(cfg, exp_dir, device):
    """The run's potentials with their fitted state (dropped columns included); decimated-DWT aware."""
    M, J, Q, wav = cfg['M'], cfg['J'], cfg['Q'], cfg.get('wav_type') or 'morlet'
    filters, fphi = return_Filters(M, J, 1, device=device, include_phi=True, wav_type=wav)
    filters_Q = return_Filters(M, J, Q, device=device, wav_type=wav)
    dwt = None
    if cfg.get('dwt_order'):
        from codes.ortho_wavelet.periodic_dwt import PeriodicDWT
        dwt = PeriodicDWT(cfg['dwt_order'], J, M, device=device)
    pots = get_1d_potentials(cfg['terms'], J, filters, Q, scalar_param=None, parallel=False, filters_Q=filters_Q,
                             filters_Phi=fphi, deduplicate_filters=bool(cfg.get('deduplicate_filters')),
                             dedup_tol=cfg.get('dedup_tol') or 1e-12, dwt=dwt)
    for name, p in pots.items():
        p.to(device)
        if hasattr(p, 'is_fitted'):
            pots[name] = type(p).load_fixed_parameters(exp_dir / 'fitted_potentials' / f'{name}.pt', p.filters,
                                                       map_location=device).to(device)
    return pots


def _pt(path):
    path = Path(path)
    return torch.load(path if path.exists() else path.with_suffix(''), map_location='cpu', weights_only=False)


def analyse(label, seed, lam=5e-4, ridge=1e-4, device=None, n_model=None, root=TURB):
    """Everything for one (label, seed). Returns a dict of numpy arrays / floats (see save)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    exp, config = find_run(label, seed, root)
    cfg = json.loads((exp / 'config.json').read_text())
    sr = root / 'saved_results'
    theta, theta_src = load_theta(config, lam, ridge, root)
    pots = rebuild(cfg, exp, device)
    # column ranges of theta per family: the fitted potentials know their true size (pruned / dropped columns
    # included), the same numbers the SDE concatenated; the assertion is the guard against any mismatch
    ends = np.cumsum([0] + [p.num_coefficients for p in pots.values()])
    ranges = {k: (int(ends[i]), int(ends[i + 1])) for i, k in enumerate(pots)}
    labels = [l for _, _, l in stat_labels(pots)]
    assert ends[-1] == theta.shape[0] == len(labels), (ends[-1], theta.shape, len(labels))

    x1, _ = build_data(int(cfg.get('subseries_len', 512)), int(cfg['M']), int(cfg['n1']), 1)
    xt = _pt(sr / 'samples' / f'{config}.pt').float()
    xt = xt.reshape(xt.shape[0], 1, -1)
    if n_model:
        xt = xt[:n_model]
    phi_d = phi_matrix(pots, x1.to(device))
    phi_m = phi_matrix(pots, xt.to(device))
    lp_d, lp_m = phi_d @ theta, phi_m @ theta                     # theta . phi  (= log p + log Z)

    # entropy bound and log Z (lower bounds, see module docstring)
    results = {config: dict(xt=xt, theta_t=_pt(sr / 'lagrange_multipliers' / f'{config}.pt'),
                            dH_t_bound=_pt(sr / 'entropy_bounds' / f'{config}.pt'),
                            t=_pt(sr / 'sampling_times' / f'{config}.pt'))}
    eb = entropy_bound(results, config, pots, check_p0=False)
    m_target = _pt(sr / 'aux_moments' / f'{config}_aux_moments.pt')['barphi_e'][-1].double().numpy()
    m_part = phi_m.mean(0)
    log_Z_lb = float(eb['H_bound'] + theta @ m_target)
    nlp_m, nlp_d = log_Z_lb - lp_m, log_Z_lb - lp_d               # -log p

    out = dict(label=label, seed=seed, config=config, theta_src=theta_src, d=int(xt.shape[-1]),
               H_bound=float(eb['H_bound']), log_Z_lb=log_Z_lb,
               H_sample=float(nlp_m.mean()), gap=float(theta @ (m_target - m_part)),
               std_model=float(nlp_m.std(ddof=1)), cross_entropy_data=float(nlp_d.mean()),
               std_data=float(nlp_d.std(ddof=1)), nlp_model=nlp_m.astype(np.float32),
               nlp_data=nlp_d.astype(np.float32), theta=theta, m_target=m_target, m_part=m_part,
               labels=np.array(labels), families=np.array(list(ranges)),
               fam_ranges=np.array(list(ranges.values())))
    for F, (a, b) in ranges.items():                              # C_F = theta_F . phi_F(x)
        out[f'C_model__{F}'] = (phi_m[:, a:b] @ theta[a:b]).astype(np.float32)
        out[f'C_data__{F}'] = (phi_d[:, a:b] @ theta[a:b]).astype(np.float32)
    # per-scale contributions of the families whose statistics sit on channels (labels '...[chJ,regK]')
    ch = np.array([int(m.group(1)) if (m := re.search(r'ch(\d+)', l)) else -1 for l in labels])
    for F, (a, b) in ranges.items():
        chans = sorted(set(ch[a:b].tolist()) - {-1})
        if chans:
            sel = [np.flatnonzero(ch[a:b] == c) + a for c in chans]
            out[f'chans__{F}'] = np.array(chans)
            out[f'Cch_model__{F}'] = np.stack([phi_m[:, s] @ theta[s] for s in sel], 1).astype(np.float32)
            out[f'Cch_data__{F}'] = np.stack([phi_d[:, s] @ theta[s] for s in sel], 1).astype(np.float32)
    out['std_over_H'] = out['std_model'] / abs(out['H_sample'])
    return out


def save(out, outdir=HERE / 'results'):
    d = Path(outdir) / out['label']
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"seed{out['seed']}.npz"
    np.savez(f, **out)
    return f


def load_all(models, seeds, outdir=HERE / 'results'):
    """{display name: [npz dicts, one per available seed]} (missing files are skipped)."""
    res = {}
    for name, label in models.items():
        runs = []
        for s in seeds:
            f = Path(outdir) / label / f'seed{s}.npz'
            if f.exists():
                z = np.load(f, allow_pickle=True)
                runs.append({k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files})
        res[name] = runs
    return res


def summary_table(res, H_gauss=None):
    """One row per model: mean +- std over the available seeds of the entropy / concentration numbers.
    H_gauss = entropy of the Gaussian with the data covariance (utils_entropy.data_gaussian_entropy(x1)): adds
    the negentropy H_gauss - H_bound, the entropy the model gives up relative to the Gaussian."""
    import pandas as pd
    rows = []
    for name, runs in res.items():
        if not runs:
            rows.append({'model': name, 'seeds': 0})
            continue
        def ms(k):
            v = np.array([r[k] for r in runs], float)
            return f'{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}'
        row = {'model': name, 'seeds': len(runs), 'H_bound': ms('H_bound')}
        if H_gauss is not None:
            hb = np.array([r['H_bound'] for r in runs], float)
            row['H_gauss (data)'] = f'{H_gauss:.2f}'
            row['negentropy H_gauss - H_bound'] = f'{(H_gauss - hb).mean():.2f} ± {(H_gauss - hb).std(ddof=1) if len(hb) > 1 else 0:.2f}'
        row.update({'H_sample (mean -log p)': ms('H_sample'), 'gap = theta.(m_target - <phi>)': ms('gap'),
                    'std(-log p) model': ms('std_model'), 'std / H': ms('std_over_H'),
                    'cross-entropy (data)': ms('cross_entropy_data'), 'std(-log p) data': ms('std_data')})
        rows.append(row)
    return pd.DataFrame(rows).set_index('model')
