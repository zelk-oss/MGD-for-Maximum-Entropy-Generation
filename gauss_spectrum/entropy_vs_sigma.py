"""Bound / true value / MGD entropy vs sigma, one curve per nt (gauss_spectrum experiment).

Plots H - H(p*) [nats] against sigma from an endpoint.csv written by sigma_sweep_summary.py:
  solid  : entropy bound H_sigma* (Eq. 30 integrated to t = 1), colour = nt (light -> dark = small -> large)
  dashed : MGD sample entropy H(gen) (same colours)
  black  : true value H(p*) (max-entropy Gaussian with the data band energies), the 0 line
  dotted : bound extrapolated to h -> 0 from the two largest nt common to all sigma (linear in 1/nt)
Left panel symlog (linear within +-1 nat), right panel linear zoom.

Usage:
  python gauss_spectrum/entropy_vs_sigma.py --csv gauss_spectrum/nt_sweep/summary/endpoint.csv --dataset turbulence \
      --out gauss_spectrum/entropy_vs_sigma_lagrangian.png --title "Lagrangian turbulence spectrum, M=256"
  python gauss_spectrum/entropy_vs_sigma.py --csv ... --zoom -15 25
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--dataset", required=True, choices=["jets", "turbulence"])
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--zoom", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="y range of the zoom panel")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    d = df[df.dataset == args.dataset]
    if d.empty:
        raise SystemExit(f"no rows for dataset {args.dataset} in {args.csv}")
    hstar = d.H_star_mean.iloc[0]
    nts = sorted(d.nt.unique())
    cmap = plt.get_cmap("Blues")
    colours = {nt: cmap(0.35 + 0.65 * i / max(1, len(nts) - 1)) for i, nt in enumerate(nts)}
    label = lambda nt: f"{nt // 1000}k" if nt < 1_000_000 else f"{nt / 1e6:g}M"

    # h -> 0: linear in 1/nt through the two largest nt that exist for every sigma (nt ratio need not be 2)
    common = [nt for nt in nts if set(d[d.nt == nt].sigma) == set(d.sigma)]
    n_a, n_b = (common[-2], common[-1]) if len(common) >= 2 else (None, None)
    sg, ex = [], []
    if n_a is not None:
        for s, g in d.groupby("sigma"):
            ya = g[g.nt == n_a].H_bound_1_mean.iloc[0]
            yb = g[g.nt == n_b].H_bound_1_mean.iloc[0]
            ua, ub = 1.0 / n_a, 1.0 / n_b
            sg.append(s)
            ex.append(yb - (yb - ya) * ub / (ub - ua) - hstar)       # value at 1/nt = 0

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, zoom in zip(axs, [False, True]):
        for nt in nts:
            g = d[d.nt == nt].sort_values("sigma")
            kw = dict(color=colours[nt], lw=1.6, marker="o", ms=5, mec="white", mew=0.8)
            ax.plot(g.sigma, g.H_bound_1_mean - hstar, label=f"bound H_σ*, nt={label(nt)}", **kw)
            ax.plot(g.sigma, g.H_gen_1_mean - hstar, ls="--", lw=1.6, color=colours[nt], marker="s", ms=5,
                    mec="white", mew=0.8)
        if n_a is not None:
            ax.plot(sg, ex, color="#d95f02", lw=1.8, ls=":", marker="D", ms=5, mec="white", mew=0.8,
                    label=f"bound, h→0 (from {label(n_a)}, {label(n_b)})")
        ax.axhline(0, color="black", lw=1.4, label="true value H(p*) (max-ent)")
        ax.set_xscale("log")
        ax.set_xlabel("σ")
        ax.set_ylabel("H − H(p*)  [nats]   (bound must stay ≤ MGD entropy ≤ 0 line)")
        ax.grid(alpha=0.2)
        if zoom:
            if args.zoom:
                ax.set_ylim(*args.zoom)
            ax.set_title("zoom (linear)")
        else:
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title("full range (symlog, linear within ±1)")
    ticks = [-100, -10, -1, 0, 1, 10, 100]
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels([str(t).replace("-", "−") for t in ticks])
    axs[0].plot([], [], ls="--", color=colours[nts[-1]], marker="s", ms=5, label="MGD sample entropy H(gen) (all nt, same colours)")
    axs[1].legend(*axs[0].get_legend_handles_labels(), fontsize=8, loc="upper left")
    fig.suptitle((args.title or args.dataset) + ": bound (solid) and MGD sample entropy H(gen) (dashed squares) vs σ; "
                 "colour = nt (light→dark = small→large)", fontsize=11)
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
