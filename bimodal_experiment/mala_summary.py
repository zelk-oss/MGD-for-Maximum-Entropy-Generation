"""Condense the MALA outputs (mala_reg.py, lam A and D) into one small file for the notebook, on Jean Zay.

Per (beta, run, start, lam rule): per-step acceptance, cloud moments E[x^a] and P(x < x_b) (float32), the exact
p_reg references, h, lam, the data moments m, and a histogram of every saved cloud snapshot (fixed bins).
The full files (cloud snapshots, traces, final state) stay on WORK.

Output: results_bim_theta_beta/mala_summary.pt (~5 MB).
Usage, from bimodal_experiment/:  python mala_summary.py
"""
from pathlib import Path

import torch

RES = Path(__file__).resolve().parent / 'results_bim_theta_beta'
DIRS = {0.1: RES / 'mala_start_beta0.1_1123/beta_0.1000/mala', 0.9: RES / 'mala_start_beta0.9_1123/beta_0.9000/mala'}
BINS = torch.linspace(-4, 4, 201)

out = []
for beta, d in DIRS.items():
    for k in (0, 1, 2):
        m = torch.load(d / f'run_{k:03d}_mgd_lamD.pt', map_location='cpu', weights_only=False)['data_moments']
        for rule, suffix in (('A', ''), ('D', '_lamD')):
            for start in ('mgd', 'gauss'):
                r = torch.load(d / f'run_{k:03d}_{start}{suffix}.pt', map_location='cpu', weights_only=False)
                cont = d / f'run_{k:03d}_{start}{suffix}_cont.pt'
                if cont.exists():                                        # continued chain: append its steps
                    c = torch.load(cont, map_location='cpu', weights_only=False)
                    r['acc_rate'] = torch.cat([r['acc_rate'], c['acc_rate']])
                    r['moments'] = torch.cat([r['moments'], c['moments'][1:]])
                    r['p_below'] = torch.cat([r['p_below'], c['p_below'][1:]])
                    r['snapshots'].update({r['steps'] + s: x for s, x in c['snapshots'].items() if s > 0})
                    r['steps'] += c['steps']
                out.append({
                    'beta': beta, 'run': k, 'start': start, 'lam_rule': rule, 'lam': r['lam'], 'h': r['h'],
                    'pilot_h_acc': r['pilot_h_acc'], 'n_walkers': r['n_walkers'], 'steps': r['steps'],
                    'acc_rate': r['acc_rate'].float(), 'moments': r['moments'].float(), 'p_below': r['p_below'].float(),
                    'exact': r['exact'], 'data_moments': m, 'theta_code': r['theta_code'], 'x_b': r['x_b'],
                    'hist_bins': BINS,
                    'hist': {s: torch.histc(x.float(), bins=len(BINS) - 1, min=float(BINS[0]), max=float(BINS[-1]))
                             for s, x in r['snapshots'].items()},
                    'source': str(d / f'run_{k:03d}_{start}{suffix}.pt'),
                })
                print(f'beta {beta} run {k} {start:5s} lam_{rule} {r["lam"]:g}: ok', flush=True)
torch.save(out, RES / 'mala_summary.pt')
print(f'saved {len(out)} entries -> {RES / "mala_summary.pt"}')
