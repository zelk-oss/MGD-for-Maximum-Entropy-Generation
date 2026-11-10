#!/usr/bin/env python3
"""Summary of the sigma sweep (launch_sigma_sweep.sh), from turbulence/logz_gauss_benchmark.py outputs
(<root>/analysis/<dataset>/<group>/logz_benchmark.csv and path_seed*.npz). Nats.
Writes <root>/summary/:
  endpoint.png  vs sigma: H_bound - H(p_t_end), and E_data[-log p] - H(p_1) with log Z exact (= KL, tests Theta_reg),
                log Z bound with theta_end = Theta_reg[-1], log Z bound with theta_end = theta_t[-1]
  path.png      H_bound(t) - H(p_t) along the transport, one curve per sigma (mean over seeds)
  endpoint.csv  mean and std over seeds of every benchmark column
  fig2b.png     MGD paper Fig. 2(b,c): H(p*), H(p_1^sigma) of the samples, bound H_sigma* vs sigma^2 (needs the
                benchmark columns H_star / H_gen_1 / H_bound_1, i.e. logz_gauss_benchmark.py from 11-09 on)
Usage: python gauss_spectrum/sigma_sweep_summary.py --root gauss_spectrum/sigma_sweep
"""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'turbulence': 'Lagrangian (turbulence spectrum)', 'jets': 'Eulerian (Modane spectrum)'}
ENDPOINT = (('Hbound_minus_Hexact_tend', r'$H_{bound} - H(p_{t_{end}})$', 'C3'),
            ('Emlogp_exact_reg_data_minus_H1', r'$E_{data}[-\log p] - H(p_1)$, exact $\log Z$ (= KL)', 'C2'),
            ('Emlogp_bound_reg_data_minus_H1', r'same, $\log Z$ bound with $\Theta_{reg}$', 'C1'),
            ('Emlogp_bound_theta_t_data_minus_H1', r'same, $\log Z$ bound with $\theta_t$', 'C4'))


def load(root):
    out = {}
    for ds in NAMES:
        for g in sorted((root / 'analysis' / ds).glob('*')):
            f = g / 'logz_benchmark.csv'
            rows = list(csv.DictReader(open(f))) if f.exists() else []
            if rows:
                paths = [dict(np.load(s)) for s in sorted(g.glob('path_seed*.npz'))]
                out.setdefault(ds, []).append(dict(sigma=float(rows[0]['sigma']), nt=int(rows[0]['nt']), rows=rows, paths=paths))
    for ds in out:
        out[ds].sort(key=lambda r: (r['sigma'], r['nt']))
    return out


def subset(data, nt):
    """Runs of one nt only (nt sweep: one fig2b per nt, so sigma and nt are not mixed in one series)."""
    out = {ds: [r for r in runs if r['nt'] == nt] for ds, runs in data.items()}
    return {ds: runs for ds, runs in out.items() if runs}


def nt_convergence(data, sd):
    """nt sweep (launch_nt_sweep.sh): at fixed sigma, entropy gaps vs the bulk step h sigma^2 = 4.5 sigma^2 / nt
    (two_phase grid, n_bulk = nt/5 on [0, 0.9]). One line per sigma. If the gap is a first-order time-discretisation
    error it is proportional to h: the printed ratios gap(nt_i)/gap(nt_{i+1}) then equal nt_{i+1}/nt_i."""
    gaps = (('Hgen_minus_Hbound', r'$H(p_1^\sigma) - H_\sigma^*$ (< 0: bound violated)'),
            ('Hstar_minus_Hgen', r'$H(p^*) - H(p_1^\sigma)$ (< 0: generated moments $\neq m$)'))
    fig, axes = plt.subplots(len(gaps), len(data), figsize=(6.5 * len(data), 4.2 * len(gaps)), squeeze=False)
    print('\nnt convergence (nats, mean over seeds); ratio = gap(previous nt) / gap(this nt), O(h) -> nt ratio')
    for i, (ds, runs) in enumerate(data.items()):
        sigmas = sorted({r['sigma'] for r in runs})
        cmap = plt.get_cmap('viridis', len(sigmas))
        for j, (k, lab) in enumerate(gaps):
            ax = axes[j, i]
            for c, s in enumerate(sigmas):
                rs = [r for r in runs if r['sigma'] == s]
                hs2 = np.array([4.5 * s ** 2 / r['nt'] for r in rs])
                g = np.array([np.nanmean(Hg - Hb) if k == 'Hgen_minus_Hbound' else np.nanmean(Hs - Hg)
                              for r in rs
                              for Hs, Hg, Hb in [(col(r['rows'], 'H_star'), col(r['rows'], 'H_gen_1'),
                                                  col(r['rows'], 'H_bound_1'))]])
                ax.plot(hs2, g, '-o', color=cmap(c), ms=5, label=rf'$\sigma$={s:g} (nt {", ".join(str(r["nt"]) for r in rs)})')
                if j == 0:
                    for a, b, ra, rb in zip(g[:-1], g[1:], rs[:-1], rs[1:]):
                        print(f'{ds:10s} sigma {s:5g}  nt {ra["nt"]:6d} -> {rb["nt"]:6d}: gap {a:+9.2f} -> {b:+9.2f}  '
                              f'ratio {a / b if b != 0 else np.nan:6.2f}  (O(h): {rb["nt"] / ra["nt"]:.2f})')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1)
            ax.set_xlabel(r'bulk step $\times\,\sigma^2$ = 4.5 $\sigma^2$/nt'); ax.set_ylabel(lab + ' (nats)', fontsize=8)
            ax.set_title(NAMES[ds], fontsize=10); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(sd / 'nt_convergence.png', dpi=150); plt.close(fig)
    print(f'wrote {sd}/nt_convergence.png')


def col(rows, k):
    return np.array([float(r[k]) for r in rows])


def fig2b(data, sd, name='fig2b.png'):
    """MGD paper Fig. 2(b,c) for the sweep. Top: H(p*) (red line, exact max-ent Gaussian), H(p_1^sigma) of the generated
    samples (blue), bound H_sigma* (black), H(data) (grey dashed), vs sigma^2. Bottom: H(p*) - H(p_1^sigma) and
    H(p*) - H_sigma* (both >= 0 in continuous time), symlog, with a sigma^-2 guide. nt under each tick: below
    sigma ~ 2.6 nt = 2000 is fixed, so h sigma^2 grows with sigma there (sigma and time step are confounded)."""
    fig, axes = plt.subplots(2, len(data), figsize=(6.5 * len(data), 8.2), squeeze=False)
    print('\nFig 2(b) at t=1 (nats, mean over seeds)\ndataset     sigma^2      nt   H(p*)  H(data)  H(gen)  H_sigma*  '
          'H*-Hgen  H*-H_sigma*  Hgen-H_sigma*')
    for i, (ds, runs) in enumerate(data.items()):
        s2 = np.array([r['sigma'] ** 2 for r in runs])
        mean = lambda k: np.array([np.nanmean(col(r['rows'], k)) for r in runs])
        Hs, Hd, Hg, Hb = mean('H_star'), mean('H_exact_1'), mean('H_gen_1'), mean('H_bound_1')
        ax = axes[0, i]
        ax.axhline(np.nanmean(Hs), color='C3', lw=1.5, label=r'$H(p^*)$ exact max-ent')
        ax.axhline(np.nanmean(Hd), color='0.5', lw=1, ls='--', label=r'$H$(data Gaussian)')
        for k, c, lab in (('H_gen_1', 'C0', r'$H(p_1^\sigma)$ generated samples'), ('H_bound_1', 'k', r'$H_\sigma^*$ bound')):
            for x, r in zip(s2, runs):
                ax.plot(np.full(len(r['rows']), x), col(r['rows'], k), '.', color=c, alpha=0.4)
            ax.plot(s2, mean(k), 'o', color=c, ms=5, label=lab)
        ax.set_ylabel('entropy (nats)'); ax.set_title(NAMES[ds], fontsize=10); ax.legend(fontsize=8, frameon=False)
        ax = axes[1, i]
        gaps = (('Hstar_minus_Hgen', 'C0', r'$H(p^*) - H(p_1^\sigma)$'), ('Hstar_minus_Hbound', 'k', r'$H(p^*) - H_\sigma^*$'))
        for k, c, lab in gaps:
            ax.plot(s2, mean(k), '-o', color=c, ms=5, label=lab)
        g0 = mean('Hstar_minus_Hgen')[0]
        if g0 > 0:
            ax.plot(s2, g0 * s2[0] / s2, ':', color='0.4', label=r'$\propto \sigma^{-2}$')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_yscale('symlog', linthresh=1); ax.set_ylabel('entropy gap (nats); < 0 violates the paper')
        ax.legend(fontsize=8, frameon=False)
        for ax in axes[:, i]:
            ax.set_xscale('log'); ax.minorticks_off()
            ax.set_xticks(s2, [f'{x:.3g}\n{r["nt"]}' for x, r in zip(s2, runs)], fontsize=7)
            ax.set_xlabel(r'$\sigma^2$  (second line: nt)')
        for x, r, a, b, c, d in zip(s2, runs, Hs, Hd, Hg, Hb):
            print(f'{ds:10s} {x:8.3g} {r["nt"]:7d} {a:7.1f} {b:8.1f} {c:7.1f} {d:9.1f} {a - c:+8.1f} {a - d:+12.1f} {c - d:+14.1f}')
    fig.tight_layout(); fig.savefig(sd / name, dpi=150); plt.close(fig)
    print(f'wrote {sd}/{name}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    data = load(args.root)
    if not data:
        print(f'nothing under {args.root}/analysis'); return
    sd = args.root / 'summary'
    sd.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(data), figsize=(6.5 * len(data), 4.6), squeeze=False)
    table = []
    for ax, (ds, runs) in zip(axes[0], data.items()):
        sig = np.array([r['sigma'] for r in runs])
        nts = sorted({r['nt'] for r in runs})
        multi = len(nts) < len(runs) and len(nts) > 1      # nt sweep: one line per nt (line style), else one line
        styles = dict(zip(nts, ('-', '--', ':', '-.'))) if multi else {}
        for k, lab, c in ENDPOINT:
            for s, r in zip(sig, runs):
                ax.plot(np.full(len(r['rows']), s), col(r['rows'], k), '.', color=c, alpha=0.5)
            for nt in (nts if multi else [None]):
                sel = [i for i, r in enumerate(runs) if nt is None or r['nt'] == nt]
                ax.plot(sig[sel], [np.nanmean(col(runs[i]['rows'], k)) for i in sel], styles.get(nt, '-'),
                        marker='o', color=c, ms=4, label=lab + (f' (nt {nt})' if multi else ''))
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1)
        ax.set_xticks(sig, [f'{s:g}' for s in sig], fontsize=8); ax.minorticks_off()
        ax.set_xlabel(r'$\sigma$' + ('' if multi else r'  (nt = max(2000, 300$\sigma^2$))'))
        ax.set_ylabel('estimate $-$ exact entropy (nats)')
        ax.set_title(NAMES[ds], fontsize=10)
        for r in runs:
            row = dict(dataset=ds, sigma=r['sigma'], nt=r['nt'], seeds=len(r['rows']))
            for k in r['rows'][0]:
                if k in ('seed', 'sigma', 'nt', 'm_source'):
                    continue
                v = col(r['rows'], k)
                row[k + '_mean'], row[k + '_std'] = float(np.nanmean(v)), float(np.nanstd(v))
            table.append(row)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(sd / 'endpoint.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(data), figsize=(6.5 * len(data), 4.6), squeeze=False)
    for ax, (ds, runs) in zip(axes[0], data.items()):
        cmap = plt.get_cmap('viridis', len(runs))
        for i, r in enumerate(runs):
            if r['paths']:
                t = r['paths'][0]['t_reg']
                err = np.mean([q['H_bound'] - q['H_exact'] for q in r['paths']], 0)
                ax.plot(1 - t, err, color=cmap(i), lw=1.2, label=rf'$\sigma$={r["sigma"]:g} (nt {r["nt"]})')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xscale('log'); ax.invert_xaxis(); ax.set_yscale('symlog', linthresh=1)
        ax.set_xlabel(r'$1 - t$  (transport runs left to right)'); ax.set_ylabel(r'$H_{bound}(t) - H(p_t)$ exact (nats)')
        ax.set_title(NAMES[ds], fontsize=10); ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(sd / 'path.png', dpi=150); plt.close(fig)

    if all('H_star' in r['rows'][0] for runs in data.values() for r in runs):
        # nt sweep: several nt per sigma -> one fig2b per nt + the convergence figure; sigma sweep: one fig2b
        multi_nt = any(len({r['nt'] for r in runs if r['sigma'] == s}) > 1
                       for runs in data.values() for s in {r['sigma'] for r in runs})
        if multi_nt:
            for nt in sorted({r['nt'] for runs in data.values() for r in runs}):
                fig2b(subset(data, nt), sd, f'fig2b_nt{nt}.png')
            nt_convergence(data, sd)
        else:
            fig2b(data, sd)

    with open(sd / 'endpoint.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    print('dataset     sigma     nt  seeds  Hbound-H(t_end)  logZbound(Theta_reg)-exact  logZbound(theta_t)-exact  '
          'E[-logp]-H: exactZ  bound(Theta_reg)  bound(theta_t)')
    for r in table:
        print(f'{r["dataset"]:10s} {r["sigma"]:6g} {r["nt"]:6d} {r["seeds"]:5d} {r["Hbound_minus_Hexact_tend_mean"]:+16.2f} '
              f'{r["logZ_bound_reg_minus_exact_reg_mean"]:+27.2f} {r["logZ_bound_theta_t_minus_exact_reg_mean"]:+25.2f} '
              f'{r["Emlogp_exact_reg_data_minus_H1_mean"]:+19.2f} {r["Emlogp_bound_reg_data_minus_H1_mean"]:+17.2f} '
              f'{r["Emlogp_bound_theta_t_data_minus_H1_mean"]:+15.2f}')
    print(f'wrote {sd}/endpoint.png, path.png, endpoint.csv')


if __name__ == '__main__':
    main()
