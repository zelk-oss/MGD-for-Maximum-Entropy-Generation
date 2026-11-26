"""Number of terminal-phase rows in a saved Theta_reg system = nodes with t after the end of the planned grid
(2026-11-25). Counted from the system's own times, not from the terminal log: a terminal step that is not in the
system (e.g. t past 1 in a smoke run, 717357) would make len(log) drop planned nodes too.
Prints N (>= 0), or -1 if the run did not put its terminal steps in the system.
Usage: python terminal_phase_test/count_terminal_rows.py <system.pt> <aux_moments/<cfg>_terminal.pt> <sampling_times file>
"""
import sys

import numpy as np
import torch

system_path, term_path, times_path = sys.argv[1:4]
term = torch.load(term_path, map_location='cpu', weights_only=False)
if not term['params'].get('in_system'):
    print(-1)
    sys.exit(0)
t_run = np.asarray(torch.load(times_path, map_location='cpu', weights_only=False), dtype=np.float64).ravel()
t_planned_end = t_run[int(term['terminal_start'])]          # last node of the planned grid
t_sys = np.asarray(torch.load(system_path, map_location='cpu', weights_only=False, mmap=True)['t'], dtype=np.float64)
n = int((t_sys > t_planned_end).sum())
print(f'planned grid ends at 1 - t = {1 - t_planned_end:.3e}; system: {len(t_sys)} nodes, {n} after it; '
      f'terminal log: {len(term["log"])} steps', file=sys.stderr)
print(n)
