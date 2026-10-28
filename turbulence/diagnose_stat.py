"""Why does one statistic stay off its target? (e.g. Scalar_psi_gaussianK[6] in mass_L6_2b_xc)

For the named statistic of a run:
  1. what it is: channel, window [lo, hi), alpha, sigmoid edge width s, data fraction inside,
     step noise nu = sqrt(2h) sigma ||g_j|| over the window width and over s at 1-t = 0.1, 1e-2, end;
     model fraction inside on the t=1 samples (hard cuts);
  2. when it goes off: p/e (walkers / target) at several 1-t, from aux_moments;
  3. whether the per-step solve can move it, from the run's conditioning log (<config>_cond.pt,
     --cond_every): share of logged steps where it is among the 3 statistics loading most on the
     weakest Gram direction, and where it is the floored statistic closest to the near-empty mask
     (margin = G_nn / floor; the statistic is switched off when margin < 1).

Usage (repo root on Jean Zay):
    python turbulence/diagnose_stat.py --label turbulence_ggd_mass_L6_2b_xc --stat 'Scalar_psi_gaussianK[6]'
"""
import argparse
import json
import re
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
from check_fit_offline import build_data                             # noqa: E402
from window_vs_step_noise import newest                              # noqa: E402

GAPS = (0.9, 0.5, 0.3, 0.1, 3e-2, 1e-2, 1e-3, 1e-4)


def main(label, seed, stat, root):
    res = root / 'saved_results'
    exps = sorted((root / 'experiments').glob(f'*/*_seed_{seed}_*{label}_2*'), key=lambda f: f.stat().st_mtime)
    exps = [x for x in exps if (res / 'aux_moments' / f'{x.name}_aux_moments.pt').exists()]
    if not exps:
        raise SystemExit(f'no run of {label} seed {seed} with aux_moments under {root}')
    exp = exps[-1]
    a = json.loads((exp / 'config.json').read_text())
    cfg = a.get('config_name', exp.name)
    M, J, Q, sigma, n1 = int(a.get('M', 128)), int(a['J']), int(a['Q']), float(a['sigma']), int(a['n1'])
    print(f'run {cfg}')

    ggd = {k: v for k, v in dict(tail_quantile=a.get('ggd_tail_quantile'),
                                 tail_alpha_bounds=a.get('ggd_tail_alpha_bounds'),
                                 tail_mass=a.get('ggd_tail_mass', False),
                                 all_masses=a.get('ggd_all_masses', False)).items() if v}
    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type=a.get('wav_type', 'morlet'))
    filters_Q = return_Filters(M, J, Q, wav_type=a.get('wav_type', 'morlet'))
    pots = get_1d_potentials(list(a['terms']), J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_phi,
                             deduplicate_filters=bool(a.get('deduplicate_filters', False)),
                             dedup_tol=float(a.get('dedup_tol', 1e-12)), ggd_kwargs=ggd)
    for name in list(pots):
        state = exp / 'fitted_potentials' / f'{name}.pt'
        if hasattr(pots[name], 'load_fixed_parameters') and state.exists():
            pots[name] = type(pots[name]).load_fixed_parameters(state, pots[name].filters, map_location='cpu')
    x1, _ = build_data(int(a.get('subseries_len', 512)), M, n1, 1)
    sizes = [pt.forward(x1[:1]).shape[1] for pt in pots.values()]
    names = [f'{nm}[{i}]' for nm, k in zip(pots, sizes) for i in range(k)]
    if stat not in names:
        raise SystemExit(f'{stat} not in this run; e.g. {names[:3]} ... {names[-3:]}')
    gidx = names.index(stat)
    pname, local = re.match(r'(.+)\[(\d+)\]$', stat).group(1), int(re.match(r'.+\[(\d+)\]$', stat).group(1))
    pt = pots[pname]
    t = np.asarray(torch.load(newest(f'{cfg}*', res / 'sampling_times'), map_location='cpu'), float).ravel()
    h, gap = np.diff(t), 1.0 - t[1:]

    # 1. what it is
    print(f'\n1. {stat} = global statistic {gidx}')
    if hasattr(pt, 'slot_window'):
        fl = int(pt.active_flat[local])
        j, k = fl % pt.J, fl // pt.J
        w, is_mass = pt.slot_window(k)
        edges = np.concatenate([[0.0], pt.cuts[j].double().numpy(), [np.inf]])
        lo, hi = edges[w], edges[w + 1]
        sw = pt.sw[j].double().numpy()
        s_edge = min(sw[m] for m in (w - 1, w) if 0 <= m < len(sw))
        f = pt.filters.to(torch.complex128)       # complex: the _imag terms carry filters * -1j
        with torch.no_grad():
            zd = torch.fft.ifft(f * torch.fft.fft(x1.double())).real[:, j].numpy()
            xs = torch.load(newest(f'{cfg}*', res / 'samples'), map_location='cpu').double()
            xs = xs[:, None] if xs.dim() == 2 else xs
            zs = torch.fft.ifft(f * torch.fft.fft(xs)).real[:, j].numpy()
        g2 = float(torch.fft.ifft(f).real.pow(2).sum(-1).reshape(-1)[j])
        inside = lambda z: float(((np.abs(z) >= lo) & (np.abs(z) < hi)).mean())
        print(f'   channel {j}, window {w} ({"mass" if is_mass else f"power alpha={float(pt.alpha[j, w]):.3f}"}), '
              f'[{lo:.3e}, {hi:.3e}), edge width s={s_edge:.2e}, live in the fit: {bool(pt.active[j, k])}')
        print(f'   fraction of |z| inside: data {inside(zd):.4f}, model t=1 {inside(zs):.4f}; '
              f'channel cuts {np.round(pt.cuts[j].double().numpy(), 6).tolist()}')
        for gg in (0.1, 1e-2):
            hv = h[np.argmin(np.abs(gap - gg))]
            nu = np.sqrt(2 * hv) * sigma * np.sqrt(g2)
            print(f'   1-t={gg:g}: step noise nu={nu:.2e}  nu/width={nu / (hi - lo if np.isfinite(hi) else np.inf):.2f}  nu/s={nu / s_edge:.1f}')
        nu = np.sqrt(2 * h[-1]) * sigma * np.sqrt(g2)
        print(f'   last step: nu={nu:.2e}  nu/width={nu / (hi - lo if np.isfinite(hi) else np.inf):.3f}  nu/s={nu / s_edge:.2f}')

    # 2. when it goes off
    aux = torch.load(res / 'aux_moments' / f'{cfg}_aux_moments.pt', map_location='cpu')
    BE, BP = aux['barphi_e'][:, gidx].double().numpy(), aux['barphi_p'][:, gidx].double().numpy()
    tr = t[:len(BE)] if len(BE) <= len(t) else np.append(t, t[-1])
    g_rows = 1.0 - tr
    print('\n2. walkers / target along the run')
    print('   ' + '  '.join(f'1-t={gg:g}: {BP[i] / BE[i]:.3f}' for gg in GAPS
                         for i in [int(np.argmin(np.abs(g_rows - gg)))]) + f'  end: {BP[-1] / BE[-1]:.3f}')

    # 3. can the solve move it
    cpath = res / 'aux_moments' / f'{cfg}_cond.pt'
    if not cpath.exists():
        print(f'\n3. no conditioning log ({cpath.name}): run had no --cond_every')
        return
    c = torch.load(cpath, map_location='cpu')
    if list(c.get('labels', [])) and c['labels'][gidx] != stat:
        print(f'   WARNING: cond-log label {c["labels"][gidx]} != {stat}')
    T = c['theta'].numpy()                                    # [k, n_live, mu_min, mu_max, top0..2, margin, margin_idx]
    kk = T[:, 0].astype(int)
    gT = 1.0 - t[np.clip(kk + 1, 0, len(t) - 1)]
    weak = (T[:, 4:7].astype(int) == gidx).any(1)
    marg = T[:, 8].astype(int) == gidx
    print(f'\n3. per-step theta solve, {len(T)} logged steps')
    for lo_g, hi_g in ((0.1, 1.01), (1e-2, 0.1), (0, 1e-2)):
        sel = (gT >= lo_g) & (gT < hi_g)
        if sel.any():
            m = T[sel & marg, 7]
            print(f'   1-t in [{lo_g:g},{hi_g:g}): in weakest-direction top 3 {weak[sel].mean():.0%} of steps; '
                  f'closest to the empty-region mask {marg[sel].mean():.0%}'
                  + (f' (margin median {np.median(m):.2g}, min {m.min():.2g}; masked if < 1)' if len(m) else '')
                  + f'; n_live median {np.median(T[sel, 1]):.0f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--stat', required=True, help="e.g. 'Scalar_psi_gaussianK[6]' (quote the brackets)")
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--root', type=Path, default=HERE, help='folder holding experiments/ and saved_results/')
    a = ap.parse_args()
    main(a.label, a.seed, a.stat, a.root)
