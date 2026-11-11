"""Per-statistic moment error at t=1 of the Eulerian (jets / Modane) runs, from the saved aux_moments only.

For every finished run of an arm (label jets_full, jets_noscalar, ...) and seed:
  e = barphi_e[-1] (target moment), p = barphi_p[-1] (walkers at t=1), rel = |p - e| / |e|,
  peak_rel = max over the run of |p(t) - e(t)| / |e(t)|  (rows 1..end).
Statistic names/order: the potentials of the run rebuilt exactly as run_experiment does (config.json), the
Scalar_*_gaussianK ones loaded from the run's fitted_potentials/ (no refit); L_6 first, then the terms in builder
order (same as turbulence/moment_error_variance.py). For Scalar statistics the (channel, window, kind) is decoded.
No data and no samples are needed (so no z-score: use turbulence/moment_error_variance.py logic for that).

Usage (repo root on Jean Zay, CPU is enough, seconds per run):
    python jets/moment_errors_jets.py --labels jets_full --seeds 500 501 502 503 504 --dump jets/moment_errors
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters_bank import return_Filters                              # noqa: E402
from potentials_builder import get_1d_potentials                     # noqa: E402

MISS = 1e-2
SCALAR = ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK',
          'Scalar_psi_gaussianK_imag', 'Scalar_morlet_gaussianK_imag')


def newest(pattern, base):
    files = sorted(base.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'nothing matches {base}/{pattern}')
    return files[-1]


def decode_scalar(pt):
    """{local index: dict(channel, window, kind, lo, hi, share)} of the active statistics of a Scalar_GGD_KRegion."""
    J, K, mw = int(pt.J), int(pt.K), pt.mass_windows
    out = {}
    for i, f in enumerate(pt.active_flat.tolist()):
        k, j = divmod(f, J)
        w, mass = (k, False) if k < K else (mw[k - K], True)
        edges = [0.0] + pt.cuts[j].tolist() + [float('inf')]
        out[i] = dict(channel=j, window=w, kind='mass' if mass else 'moment', lo=edges[w], hi=edges[w + 1],
                      share=float(pt.pi[j, w]))
    return out


def analyse(label, seed, root):
    res = root / 'saved_results'
    exps = sorted((root / 'experiments').glob(f'*/*_seed_{seed}_*_{label}_2*'), key=lambda f: f.stat().st_mtime)
    exps = [x for x in exps if (res / 'aux_moments' / f'{x.name}_aux_moments.pt').exists()]
    if not exps:
        raise FileNotFoundError(f'no run of {label} seed {seed} with aux_moments under {root}')
    exp = exps[-1]
    a = json.loads((exp / 'config.json').read_text())
    cfg = a.get('config_name', exp.name)
    M, J, Q, terms = int(a['M']), int(a['J']), int(a['Q']), list(a['terms'])
    print(f'\n=== {label} seed {seed}: {cfg}\n  n1={a["n1"]} M={M} J={J} Q={Q} nt={a["nt"]} sigma={a["sigma"]} terms={terms}')

    aux = torch.load(res / 'aux_moments' / f'{cfg}_aux_moments.pt', map_location='cpu')
    BE, BP = aux['barphi_e'].double().numpy(), aux['barphi_p'].double().numpy()
    e, p = BE[-1], BP[-1]
    try:
        t = np.asarray(torch.load(newest(f'{cfg}*', res / 'sampling_times'), map_location='cpu'), float).ravel()
    except Exception:
        t = None

    filters, filters_phi = return_Filters(M, J, 1, include_phi=True)
    ggd = {k: v for k, v in dict(tail_quantile=a.get('ggd_tail_quantile'), num_regions=a.get('ggd_num_regions'),
                                 tail_alpha_bounds=a.get('ggd_tail_alpha_bounds'),
                                 tail_mass=a.get('ggd_tail_mass') or None,
                                 all_masses=a.get('ggd_all_masses') or None,
                                 boundary_method=a.get('ggd_boundary_method')).items() if v is not None}
    pots = get_1d_potentials(terms, J, filters, Q, filters_Q=None, filters_Phi=filters_phi, scalar_param=None,
                             parallel=False, deduplicate_filters=bool(a.get('deduplicate_filters', False)),
                             dedup_tol=float(a.get('dedup_tol', 1e-12)), ggd_kwargs=ggd)
    for name in list(pots):                      # the run's fitted states (incl. drops), no refit
        state = exp / 'fitted_potentials' / f'{name}.pt'
        if hasattr(pots[name], 'load_fixed_parameters') and state.exists():
            pots[name] = type(pots[name]).load_fixed_parameters(state, pots[name].filters, map_location='cpu')

    x0 = torch.randn(1, 1, M)
    sizes = [int(pt.forward(x0).shape[1]) for pt in pots.values()]
    if sum(sizes) != len(e):
        raise SystemExit(f'{sum(sizes)} statistics rebuilt vs {len(e)} in aux_moments: terms/order mismatch')
    meta = []
    for (nm, pt), n in zip(pots.items(), sizes):
        dec = decode_scalar(pt) if nm in SCALAR else {}
        for i in range(n):
            meta.append(dict(name=f'{nm}[{i}]', family=nm, **dec.get(i, {})))

    d = (p - e) / np.abs(e).clip(1e-300)
    d_t = np.abs((BP - BE) / np.abs(BE).clip(1e-300))
    jp = d_t[1:].argmax(0) + 1
    return dict(label=label, seed=seed, cfg=cfg, a=a, meta=meta, e=e, p=p, d=d, rel=np.abs(d),
                peak=d_t[jp, np.arange(d_t.shape[1])], peak_step=jp,
                peak_gap=(1.0 - t[np.minimum(jp, len(t) - 1)]) if t is not None else np.full(len(e), np.nan))


def report(r):
    rel, fam = r['rel'], np.array([m['family'] for m in r['meta']])
    print(f'  {len(rel)} statistics at t=1; miss = rel > {MISS:g}')
    print(f'  {"family":<40}{"n":>5}{"median":>10}{"p90":>10}{"max":>10}{">1%":>6}{">0.1%":>7}')
    for f in dict.fromkeys(fam):
        x = rel[fam == f]
        print(f'  {f:<40}{len(x):5d}{np.median(x):10.1e}{np.quantile(x, .9):10.1e}{x.max():10.1e}{(x > MISS).sum():6d}{(x > 1e-3).sum():7d}')
    x = rel
    print(f'  {"ALL":<40}{len(x):5d}{np.median(x):10.1e}{np.quantile(x, .9):10.1e}{x.max():10.1e}{(x > MISS).sum():6d}{(x > 1e-3).sum():7d}')
    print('  12 worst:')
    for i in np.argsort(-rel)[:12]:
        m = r['meta'][i]
        where = (f' ch {m["channel"]} win {m["window"]} {m["kind"]} [{m["lo"]:.3g}, {m["hi"]:.3g}) share {m["share"]:.2%}'
                 if 'channel' in m else '')
        print(f'    {m["name"]:<42}rel {rel[i]:.2e} signed {r["d"][i]:+.1e} peak {r["peak"][i]:.2e}{where}')


def dump_csv(r, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{r['cfg']}.csv"
    cols = ['index', 'name', 'family', 'channel', 'window', 'kind', 'lo', 'hi', 'share', 'target_e', 'generated_p',
            'd_signed', 'rel', 'peak_rel', 'peak_step', 'peak_at_1mt']
    with open(f, 'w') as fh:
        fh.write(f"# {r['cfg']}  label={r['label']} seed={r['seed']} nt={r['a']['nt']} sigma={r['a']['sigma']}  rel=|p-e|/|e| at t=1\n")
        fh.write(','.join(cols) + '\n')
        for i, m in enumerate(r['meta']):
            row = [i, m['name'], m['family'], m.get('channel', ''), m.get('window', ''), m.get('kind', ''),
                   m.get('lo', ''), m.get('hi', ''), m.get('share', ''), r['e'][i], r['p'][i], r['d'][i], r['rel'][i],
                   r['peak'][i], int(r['peak_step'][i]), r['peak_gap'][i]]
            fh.write(','.join(f'{v:.6e}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + '\n')
    print(f'  wrote {f} ({len(r["meta"])} statistics)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', default=['jets_full'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[500, 501, 502, 503, 504])
    ap.add_argument('--root', type=Path, default=HERE, help='folder holding experiments/ and saved_results/ (default: jets/)')
    ap.add_argument('--dump', type=Path, default=None, help='folder for one CSV per run (all statistics)')
    a = ap.parse_args()
    for lab in a.labels:
        for s in a.seeds:
            try:
                r = analyse(lab, s, a.root)
            except Exception as ex:                       # keep going: report which run failed
                print(f'\n=== {lab} seed {s}: FAILED: {type(ex).__name__}: {ex}')
                continue
            report(r)
            if a.dump:
                dump_csv(r, a.dump)
