"""Shared helpers of the cosmic-web analysis scripts (logp_cw.py, moment_check_cw.py).

A run = the experiments/<group>/<config>/config.json written by run_SDE.py. From it we rebuild
  * the data x1  (same loader and standardisation as run_SDE.py),
  * the potentials of the run (2D potentials have no fit(): phi is just potential.forward(x), so phi here is
    bit-for-bit the phi of the SDE and Theta_reg . phi is the log p of the model),
  * the generated samples saved_results/samples/<config>.pt.
phi (n, r) of the data and of the samples is cached in results/phi/<config>.pt (a few MB), so the log p script and
the moment check never compute it twice.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
# where experiments/, saved_results/, results/, figures/ live (CW_ROOT: a scratch copy for smoke tests)
CW = Path(os.environ.get('CW_ROOT', CODE))
for p in (CODE, ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEEDS = [900, 901, 902, 903, 904]
RESULTS = CW / 'results'
FIGURES = CW / 'figures'


def find_runs(arm, seeds=SEEDS, cw=CW):
    """{seed: (exp_dir, cfg)} of the runs with --label <arm> (newest if a seed was run twice)."""
    found = {}
    for f in sorted((cw / 'experiments').glob('*/*/config.json'), key=lambda q: q.stat().st_mtime):
        cfg = json.loads(f.read_text())
        if cfg.get('label') == arm and cfg.get('seed') in seeds:
            found[int(cfg['seed'])] = (f.parent, cfg)
    missing = [s for s in seeds if s not in found]
    assert not missing, f'arm {arm!r}: no run for seeds {missing} under {cw / "experiments"}'
    return found


def load_x1(cfg, data_path=None):
    """The n1 standardised maps (n1, 1, M, M) of the run."""
    from run_SDE import load_quijote_2d, smoke_data
    from codes.utils import normalize
    if cfg.get('smoke_data'):                      # synthetic maps of the smoke tests (run_SDE.py --smoke_data)
        return normalize(smoke_data(cfg['n1'], cfg['smoke_size']))
    data = load_quijote_2d(data_path or cfg['data_path'], fact=cfg['fact'], apply_log=not cfg['data_is_log'])
    return normalize(data[:cfg['n1']])


def load_samples(cfg, cw=CW):
    M = int(cfg['M'])
    return torch.load(cw / 'saved_results' / 'samples' / f"{cfg['config_name']}.pt", map_location='cpu').reshape(-1, 1, M, M).float()


def build_potentials(cfg, device):
    from codes.filters_bank import return_Filters
    from codes.potentials_builder import get_2d_potentials
    filters = return_Filters(int(cfg['M']), cfg['J'], cfg['Q'], L=cfg['L'], device=device)
    return get_2d_potentials(cfg['terms'], cfg['J'], cfg['L'], filters, parallel=False)


def family_slices(pots):
    """{family: (start, width)} in the column order of Theta (the dict order of the potentials, as in the SDE)."""
    out, a = {}, 0
    for k, p in pots.items():
        out[k] = (a, int(p.num_coefficients))
        a += int(p.num_coefficients)
    return out


@torch.no_grad()
def phi_matrix(pots, x, chunk=10, device='cpu'):
    """(n, r) float64: every statistic of every potential on the maps x (n, 1, M, M), `chunk` maps at a time."""
    rows = []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk].to(device)
        ph = []
        for p in pots.values():
            y = p.forward(xb)
            ph.append(y.real if torch.is_complex(y) else y)
        rows.append(torch.cat(ph, 1).double().cpu())
    return torch.cat(rows).numpy()


def get_phi(arm, seed, exp_dir, cfg, device, chunk=10, data_path=None, cache=True):
    """phi of the data and of the generated samples of one run: (phi_data (n1, r), phi_gen (n, r), families)."""
    f = RESULTS / 'phi' / f"{cfg['config_name']}.pt"
    pots = build_potentials(cfg, device)
    fam = family_slices(pots)
    if cache and f.exists():
        d = torch.load(f, map_location='cpu', weights_only=False)
        return d['phi_data'], d['phi_gen'], fam
    phi_d = phi_matrix(pots, load_x1(cfg, data_path), chunk, device)
    phi_g = phi_matrix(pots, load_samples(cfg), chunk, device)
    assert phi_d.shape[1] == phi_g.shape[1] == sum(w for _, w in fam.values()), (phi_d.shape, fam)
    if cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'phi_data': phi_d, 'phi_gen': phi_g, 'families': list(fam)}, f)
    return phi_d, phi_g, fam
