"""Small extract for explore_theta_tail_m1024.ipynb (2026-11-25): theta_t of the run and Theta_reg (all nodes = A,
without the terminal-phase nodes = B) along time, for a few coefficients, and the final values of ALL coefficients,
for every seed of one config. Runs on Jean Zay (theta files are 40-90 MB each); the output npz is a few MB.

Selection: per family, the coefficient with the largest |Theta_A(1)| * sd(phi), sd = std of the statistic over the final
walkers of the first seed (diagonal of the system's Sigma at its last node): the coefficient whose term moves log p most.

Usage (repo root, Jean Zay):
  python terminal_phase_test/extract_theta_tail.py --configs C900 C901 C902 --reg_dir <system folder>
      --theta_reg_dir terminal_phase_test/theta_reg_insys_m1024 --lam 1e-4 --names <moment csv> --out <file.npz>
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'turbulence' / 'saved_results'

ap = argparse.ArgumentParser()
ap.add_argument('--configs', nargs='+', required=True, help='one config name per seed')
ap.add_argument('--reg_dir', type=Path, required=True, help='folder of the saved systems (<config>.pt)')
ap.add_argument('--theta_reg_dir', type=Path, required=True)
ap.add_argument('--lam', type=float, default=1e-4)
ap.add_argument('--ridge', type=float, default=1e-4)
ap.add_argument('--names', type=Path, required=True, help="csv with a 'name' column, one row per statistic")
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()

with open(args.names) as fh:
    names = [row['name'] for row in csv.DictReader(line for line in fh if not line.startswith('#'))]
family = np.array([re.sub(r'\[\d+\]$', '', n) for n in names])


def theta_t_of(cfg):
    th = torch.load(RES / 'lagrange_multipliers' / f'{cfg}.pt', map_location='cpu', weights_only=False)
    th = th['theta_t'] if isinstance(th, dict) else th
    return torch.as_tensor(th).double().numpy()


out = dict(names=np.array(names), family=family)
sel = None
for s, cfg in enumerate(args.configs):
    t_run = np.asarray(torch.load(RES / 'sampling_times' / f'{cfg}.pt', map_location='cpu', weights_only=False),
                       float).ravel()
    th = theta_t_of(cfg)
    if th.shape[0] == len(t_run):
        t_th = t_run
    elif th.shape[0] == len(t_run) - 1:
        t_th = t_run[1:]                                   # one theta per step, at the end of the step
    else:
        raise SystemExit(f'{cfg}: {th.shape[0]} theta_t rows vs {len(t_run)} times: alignment unknown, stop')
    term = torch.load(RES / 'aux_moments' / f'{cfg}_terminal.pt', map_location='cpu', weights_only=False)
    k0 = int(term['terminal_start'])
    stem = f'lam{args.lam:g}_ridge{args.ridge:g}_moment_mask'
    A = torch.load(args.theta_reg_dir / cfg / f'{stem}.pt', map_location='cpu', weights_only=False)
    Bf = sorted((args.theta_reg_dir / cfg).glob(f'{stem}_droplast*.pt'))
    assert len(Bf) == 1, f'{cfg}: expected one droplast file, found {Bf}'
    B = torch.load(Bf[0], map_location='cpu', weights_only=False)
    ThA, ThB = A['Theta_reg'].double().numpy(), B['Theta_reg'].double().numpy()
    if s == 0:
        system = torch.load(args.reg_dir / f'{cfg}.pt', map_location='cpu', weights_only=False, mmap=True)
        S = system['Sigma'][len(system['t']) - 1]
        sd = torch.sqrt(torch.clamp(torch.diagonal(S).double(), min=0)).numpy()
        assert len(sd) == len(names) == ThA.shape[1], (len(sd), len(names), ThA.shape)
        score = np.abs(ThA[-1]) * sd
        sel = np.array([int(np.flatnonzero(family == f)[np.argmax(score[family == f])]) for f in dict.fromkeys(family)])
        out.update(sd=sd, sel=sel)
    out.update({f's{s}_t_run': t_run, f's{s}_t_theta': t_th, f's{s}_theta_t_sel': th[:, sel].astype(np.float32),
                f's{s}_theta_t_final': th[-1], f's{s}_k0': k0, f's{s}_t_planned_end': t_run[k0],
                f's{s}_tA': np.asarray(A['t_reg'], float), f's{s}_ThA_sel': ThA[:, sel].astype(np.float32),
                f's{s}_ThA_final': ThA[-1],
                f's{s}_tB': np.asarray(B['t_reg'], float), f's{s}_ThB_sel': ThB[:, sel].astype(np.float32),
                f's{s}_ThB_final': ThB[-1], f's{s}_config': cfg})
    print(f'seed {s}: {cfg[-60:]}\n  theta_t {th.shape}, run times {len(t_run)}, planned steps {k0}, '
          f'Theta_A {ThA.shape}, Theta_B {ThB.shape} ({Bf[0].name})')
out['n_seeds'] = len(args.configs)
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, **out)
print(f'selected: {[names[i] for i in sel]}\n-> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)')
