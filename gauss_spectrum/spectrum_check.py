#!/usr/bin/env python3
"""Where is the entropy of the MGD samples missing? (nt sweep, 2026-11-11)

Lagrangian Gaussian: H(p_1^sigma) sits ~2.4 nats below H(p*) for every nt (10k-50k), n1 (1e4, 1e5) and sigma^2 (0.25-4);
Eulerian: within 0.2 nats. Everything is Gaussian, stationary and diagonal in the ortho Fourier basis, so frequency by
frequency
    H(gen) - H(p*) = sum_w 1/2 log(C_gen(w) / C*(w))          (exact; C = power spectrum, C* = max-ent spectrum)
Per run (one nt, all sigma, both datasets) this prints / saves:
  - total deficit, and how much of it comes from modes that Langevin cannot relax in unit time: OU rate of mode w is
    sigma^2 / C*(w), so hypothesis 1 = deficit concentrated where sigma^2 / C* < 1 (large-variance, low-frequency modes);
  - band moments of the samples vs the target at t = 1 and at t_end = 1 - gap_end (the run stops at t_end and matches
    m(t_end) = sin^2 m(1) + cos^2 m_white) and H(p*) for m(t_end): hypothesis 2 = the samples target m(t_end), not m(1).
Outputs in --outdir: spectrum_check.png, spectrum_check_<dataset>_s<sigma>.npz.
Usage (JZ, repo root, CPU is enough): python gauss_spectrum/spectrum_check.py --root gauss_spectrum/nt_sweep --nt 50000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'turbulence'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import logp_gauss_scat2 as lg                                                    # noqa: E402
from logz_gauss_benchmark import max_ent_from_spectrum                           # noqa: E402

NAMES = {'turbulence': 'Lagrangian', 'jets': 'Eulerian'}


def runs(root, ds, nt):
    """(sigma, seed_dir, cfg) for every run of dataset ds with this nt."""
    out = []
    for g in sorted((root / ds / 'experiments').iterdir()):
        for d in sorted(g.iterdir()) if g.is_dir() else []:
            if (d / 'config.json').exists():
                cfg = json.loads((d / 'config.json').read_text()); cfg = cfg.get('args', cfg)
                if int(cfg['nt']) == nt:
                    out.append((float(cfg['sigma']), d, cfg))
    return sorted(out, key=lambda r: r[0])


def spectrum(x, M):
    x = x.reshape(-1, M).double()
    x = x - x.mean(0, keepdim=True)                                              # as compute_gaussian_entropy
    return (torch.fft.fft(x, norm='ortho').abs() ** 2).mean(0).numpy()


def draw_dataset(axes_row, ds, nt, w, C_data, Cs, items):
    """One row of the figure. items = [(sigma, C_gen, dH)] sorted by sigma. Shared by the compute and --plot_only paths."""
    M = len(C_data)
    ax = axes_row[0]
    ax.loglog(w, C_data[w], 'k', lw=2, label='data')
    ax.loglog(w, Cs[w], 'C3--', lw=2, label=r'max-ent $p^*$')
    cmap = plt.get_cmap('viridis', len(items))
    for k, (s, C_gen, dH) in enumerate(items):
        rate = s ** 2 / Cs
        tot = float(dH.sum())
        ax.loglog(w, C_gen[w], color=cmap(k), lw=0.8, label=rf'samples $\sigma$={s:g}')
        axes_row[1].plot(w, np.cumsum(dH[w] + dH[(-w) % M] * (w < M // 2)), color=cmap(k),
                         label=rf'$\sigma$={s:g} (total {tot:+.1f})')
        axes_row[2].semilogx(rate[w], dH[w], '.', color=cmap(k), ms=3, label=rf'$\sigma$={s:g}')
    ax.set_title(f'{NAMES[ds]}: power spectra (nt {nt})'); ax.set_xlabel('frequency index'); ax.legend(fontsize=7)
    axes_row[1].axhline(0, color='0.6', lw=0.8)
    axes_row[1].set_xscale('log'); axes_row[1].set_xlabel('frequency index (low -> high)')
    axes_row[1].set_ylabel(r'cumulative $\frac{1}{2}\sum \log(C_{gen}/C^*)$ (nats)')
    axes_row[1].set_title('where the entropy difference H(gen) - H(p*) builds up'); axes_row[1].legend(fontsize=7)
    axes_row[2].axhline(0, color='0.6', lw=0.8); axes_row[2].axvline(1, color='0.6', ls=':')
    axes_row[2].set_xlabel(r'relaxation rate $\sigma^2/C^*(\omega)$ (< 1: mode not relaxed in unit time)')
    axes_row[2].set_ylabel(r'$\frac{1}{2}\log(C_{gen}/C^*)$ per frequency (nats)')
    axes_row[2].set_title('per-frequency difference vs relaxation rate'); axes_row[2].legend(fontsize=7)


def plot_only(nt, out):
    """Redraw spectrum_check.png from the spectrum_check_<dataset>_s<sigma>.npz files in out (no recomputation)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for i, ds in enumerate(('turbulence', 'jets')):
        zs = [dict(np.load(f)) for f in sorted(out.glob(f'spectrum_check_{ds}_s*.npz'))]
        if not zs:
            print(f'{ds}: no spectrum_check_{ds}_s*.npz in {out}'); continue
        zs.sort(key=lambda z: float(z['sigma']))
        M = len(zs[0]['C_data'])
        draw_dataset(axes[i], ds, nt, np.arange(1, M // 2 + 1), zs[0]['C_data'], zs[0]['C_star'],
                     [(float(z['sigma']), z['C_gen'], z['dH']) for z in zs])
        print(f'{ds}: {len(zs)} sigma values from npz')
    fig.tight_layout(); fig.savefig(out / 'spectrum_check.png', dpi=130); plt.close(fig)
    print(f'wrote {out}/spectrum_check.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--nt', type=int, default=50000)
    p.add_argument('--outdir', type=Path, default=None)
    p.add_argument('--plot_only', action='store_true', help='redraw the figure from the saved npz, no recomputation')
    a = p.parse_args()
    out = a.outdir or a.root / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        plot_only(a.nt, out); return
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    print('dataset     sigma  H(gen)-H*  from modes sigma^2/C*<1  <10 | max |m_gen/m(1)-1|  max |m_gen/m(t_end)-1| '
          '| H*(m(t_end))-H*(m(1))  H(gen)-H*(m(t_end)) | H*(m_gen)-H*(m(1))  H(gen)-H*(m_gen)')
    for i, ds in enumerate(('turbulence', 'jets')):
        rs = runs(a.root, ds, a.nt)
        if not rs:
            print(f'{ds}: no runs with nt {a.nt}'); continue
        cfg0, d0 = rs[0][2], rs[0][1]
        M = int(cfg0['M'])
        psi2 = lg.filters_of(lg.build_pots(cfg0, 'cpu'))
        C_data = spectrum(lg.load_data_x1(cfg0, 'cpu', d0).cpu(), M)
        st1 = max_ent_from_spectrum(C_data, psi2, M)
        Cs = st1['spectrum_star']
        m1 = psi2 @ C_data / M
        w = np.arange(1, M // 2 + 1)
        items = []
        for k, (s, d, cfg) in enumerate(rs):
            ge = float(cfg.get('gap_end') or 0.0)
            a_, b_ = np.cos(np.pi * (1 - ge) / 2), np.sin(np.pi * (1 - ge) / 2)
            C_tend = b_ ** 2 * C_data + a_ ** 2                                  # interpolant spectrum at t_end
            st_e = max_ent_from_spectrum(C_tend, psi2, M)
            m_e = psi2 @ C_tend / M
            C_gen = spectrum(lg.load_samples(a.root / ds, d.name), M)
            m_gen = psi2 @ C_gen / M
            st_g = max_ent_from_spectrum(C_gen, psi2, M)                         # max-ent with the GENERATED moments
            dH = 0.5 * np.log(C_gen / Cs)                                        # per-frequency H(gen) - H(p*)
            rate = s ** 2 / Cs
            tot = float(dH.sum())
            print(f'{NAMES[ds]:10s} {s:5g} {tot:+10.2f} {dH[rate < 1].sum():+24.2f} {dH[rate < 10].sum():+6.2f} | '
                  f'{np.max(np.abs(m_gen / m1 - 1)):18.1e} {np.max(np.abs(m_gen / m_e - 1)):23.1e} | '
                  f'{st_e["H_star"] - st1["H_star"]:+22.3f} {0.5 * np.log(C_gen / st_e["spectrum_star"]).sum():+20.2f} | '
                  f'{st_g["H_star"] - st1["H_star"]:+21.3f} {0.5 * np.log(C_gen / st_g["spectrum_star"]).sum():+19.3f}')
            np.savez(out / f'spectrum_check_{ds}_s{s:g}.npz', sigma=s, C_data=C_data, C_star=Cs, C_gen=C_gen,
                     C_star_tend=st_e['spectrum_star'], C_star_gen=st_g['spectrum_star'], m1=m1, m_tend=m_e,
                     m_gen=m_gen, dH=dH, gap_end=ge)
            items.append((s, C_gen, dH))
        draw_dataset(axes[i], ds, a.nt, w, C_data, Cs, items)
    fig.tight_layout(); fig.savefig(out / 'spectrum_check.png', dpi=130); plt.close(fig)
    print(f'wrote {out}/spectrum_check.png and spectrum_check_*.npz')


if __name__ == '__main__':
    main()
