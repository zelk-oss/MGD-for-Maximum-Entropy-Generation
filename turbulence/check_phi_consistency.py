"""Are the moments reported as matched in-run the moments of the saved samples?

Two measures of moment matching disagree on turbulence_final (2026-10-20):
  in-run   barphi_p[-1] = compute_moments(final walkers).mean(0) vs barphi_e[-1]
           (saved_results/aux_moments/<config>_aux_moments.pt): median own-scale error ~1e-8
  post-hoc _phi(saved xt[:1000]) with potentials rebuilt from fitted_potentials/ vs
           barphi_e[-1] (compare_entropy_methods.py): ||m_target - m_particles|| / ||m_target||
           = 12% (seed 900), 28% (seed 901)
They should agree if xt is the final walker set and the rebuilt potentials are the run's.

For each run this prints, per family of statistics:
  p_run   = barphi_p[-1]                    (in-run walker moments)
  e       = barphi_e[-1]                    (target)
  p_10k   = _phi(xt).mean(0), all walkers   (post-hoc, rebuilt potentials)
  p_1k    = same on the first 1000 walkers  (the entropy handoff's setting)
and the errors |a - b| / max_t |barphi_e| (own scale, as check_moments.plot_moment_matching),
plus the L2 relative errors the handoff used.

Reading:
  p_10k == p_run (own-scale ~0)  -> saved xt + rebuilt potentials reproduce the in-run moments;
                                    the 12-28% was subsampling noise if p_1k is far and p_10k close.
  p_10k != p_run                 -> the in-run report is not about the samples we analyse:
                                    the family / column table says where they diverge.

Usage (from the repo root on Jean Zay; GPU node or login node, it chunks over walkers):
    python turbulence/check_phi_consistency.py
    python turbulence/check_phi_consistency.py --seed 901 --labels turbulence_final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # turbulence/
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import load_results                                                  # noqa: E402
from compare_entropy_methods import (                                           # noqa: E402
    rebuild_potentials, load_fitted_potentials, phi_mean_over_samples,
)


def find_config(label, seed):
    """Newest saved sample file for (label, seed). '_2' = start of the timestamp, so
    'turbulence_final' does not also match 'turbulence_final_noScat4'."""
    files = sorted((HERE / 'saved_results' / 'samples').glob(f'*_seed_{seed}_*_{label}_2*.pt'),
                   key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'no saved_results/samples/*_seed_{seed}_*_{label}_2*.pt')
    if len(files) > 1:
        print(f'  [{label}] {len(files)} matches, using the newest')
    return files[-1].stem


def own(a, b, scale):
    return (a - b).abs() / scale


def summary(x):
    x = x.numpy()
    return f'median {np.median(x):9.2e}  max {x.max():9.2e}'


def check(label, seed, device, chunk):
    config = find_config(label, seed)
    print(f'\n{"=" * 100}\n{label}  seed {seed}\n  {config}')
    exp_dirs = list((HERE / 'experiments').glob(f'*/{config}'))
    assert exp_dirs, f'no experiments/*/{config}'
    exp_dir = exp_dirs[0]
    cfg = json.loads((exp_dir / 'config.json').read_text())
    cfg = cfg.get('args', cfg)

    aux = torch.load(HERE / 'saved_results' / 'aux_moments' / f'{config}_aux_moments.pt',
                     map_location='cpu', weights_only=False)
    E, P = aux['barphi_e'].double(), aux['barphi_p'].double()
    e, p_run = E[-1], P[-1]
    scale = E.abs().max(0).values.clamp_min(1e-300)          # own scale, whole-run max |barphi_e|

    xt = load_results(HERE, config)[0]
    print(f'  xt {tuple(xt.shape)} {xt.dtype} | aux: {E.shape[0]} rows x {E.shape[1]} moments')

    pots = rebuild_potentials(cfg, device)
    pots = load_fitted_potentials(pots, exp_dir / 'fitted_potentials', device)
    fams = [(name, int(p.num_coefficients)) for name, p in pots.items()]
    r_post = sum(n for _, n in fams)
    print('  rebuilt potentials: ' + ', '.join(f'{n} {k}' for k, n in fams) + f'  -> r = {r_post}')
    if r_post != E.shape[1]:
        print(f'  !! rebuilt r = {r_post} != in-run r = {E.shape[1]}: the rebuilt potentials are NOT the '
              f"run's (column layout differs). Stopping here for this run.")
        return

    p_10k = phi_mean_over_samples(xt, pots, chunk, device)
    p_1k = phi_mean_over_samples(xt[:1000], pots, chunk, device)

    print('\n  L2 relative errors  ||a - b|| / ||e||  (the entropy handoff measure)')
    ne = e.norm()
    for name, a, b in (('p_run - e', p_run, e), ('p_10k - e', p_10k, e),
                       ('p_1k  - e', p_1k, e), ('p_10k - p_run', p_10k, p_run),
                       ('p_1k  - p_run', p_1k, p_run)):
        print(f'    {name:15s} {float((a - b).norm() / ne):.3e}')

    print('\n  per family, own-scale error |a - b| / max_t|e|   and share of ||e||^2')
    print(f'    {"family":34s} {"n":>4}  {"share":>6}  {"p_10k vs p_run":>30}  {"p_run vs e":>30}  '
          f'{"p_10k vs e":>30}')
    off = 0
    for name, n in fams:
        s = slice(off, off + n)
        share = float((e[s] ** 2).sum() / (ne ** 2))
        print(f'    {name:34s} {n:4d}  {share:6.1%}  {summary(own(p_10k[s], p_run[s], scale[s])):>30}  '
              f'{summary(own(p_run[s], e[s], scale[s])):>30}  {summary(own(p_10k[s], e[s], scale[s])):>30}')
        off += n

    labels = [f'{name}[{i}]' for name, n in fams for i in range(n)]
    d = own(p_10k, p_run, scale)
    order = torch.argsort(d, descending=True)[:12]
    print('\n  top 12 columns by |p_10k - p_run| (own scale)')
    print(f'    {"col":>4} {"name":34s} {"e":>11} {"p_run":>11} {"p_10k":>11} {"p_1k":>11} {"own diff":>9}')
    for i in order.tolist():
        print(f'    {i:4d} {labels[i]:34s} {e[i].item():11.4e} {p_run[i].item():11.4e} '
              f'{p_10k[i].item():11.4e} {p_1k[i].item():11.4e} {d[i].item():9.2e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--labels', nargs='+', default=['turbulence_final_noScat4', 'turbulence_final'])
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    for lab in a.labels:
        check(lab, a.seed, a.device, a.chunk)
