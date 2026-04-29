# MGD: Moment Guided Diffusion for Maximum Entropy Generation

[![arXiv](https://img.shields.io/badge/arXiv-2602.17211-b31b1b.svg)](https://arxiv.org/abs/2602.17211)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Official implementation of the paper:

> **MGD: Moment Guided Diffusion for Maximum Entropy Generation**  
> Etienne Lempereur¹, Nathanaël Cuvelle–Magar¹, Florentin Coeurdoux², Stéphane Mallat³⁴, Eric Vanden-Eijnden⁵⁶  
> ¹ ENS / Université PSL — ² Capital Fund Management — ³ Collège de France — ⁴ Flatiron Institute — ⁵ NYU Courant Institute ⁶ ML Lab CFM  
> *arXiv, February 2026*

---

## Overview

Generating samples from limited information is a fundamental challenge across scientific domains. **Moment Guided Diffusion (MGD)** is an algorithm to sample from a maximum entropy distribution estimated over data that is numerically accessible in high dimension. It bridges two previously separate paradigms:

| Approach | Input | Max-entropy guarantee | Moment control | Sampling |
|---|---|---|---|---|
| Classical max-entropy (MCMC) | Moments **m** | ✅ | ✅ | Equilibrium (slow) |
| Diffusion / Flow Matching | Dataset **(xᵢ)** | ❌ | ❌ | Non-equilibrium (fast) |
| **MGD (ours)** | **Dataset (xᵢ)** | ✅ | ✅ | **Non-equilibrium (guided)** |

MGD combines the principled uncertainty quantification of maximum entropy theory with the scalable, finite-time transport of stochastic interpolants. It solves a McKean–Vlasov SDE that steers moment constraints toward prescribed target values, avoiding the exponential slowdown that plagues classical MCMC and Langevin dynamics in high dimensions.

---

## Applications

MGD is demonstrated on three families of high-dimensional multiscale processes, all conditioned via **wavelet scattering moments**:

📈 Financial Time Series (S&P 500)
🌊 Turbulent Flows
🌌 Cosmological Fields

---

## Repository Structure

```
.
├── code/           # Core MGD implementation (Python)
├── data/           # Example datasets (financial, turbulence, cosmology)
├── notebooks/      # Jupyter notebooks reproducing paper experiments
├── saved_results/  # Pre-computed results for quick reproduction
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy scipy matplotlib
```

> Additional dependencies may be listed inside individual notebooks.

### Running the Notebooks

The `notebooks/` directory contains Jupyter notebooks corresponding to the experiments in the paper. Launch them with:

```bash
jupyter notebook notebooks/
```

Key notebooks include:
- Convergence verification 
- Financial time series generation
- Turbulent flow generation
- Cosmological field generation and entropy estimation

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{lempereur2026mgd,
  title   = {{MGD}: Moment Guided Diffusion for Maximum Entropy Generation},
  author  = {Lempereur, Etienne and Cuvelle--Magar, Nathanaël and Coeurdoux, Florentin and Mallat, Stéphane and Vanden-Eijnden, Eric},
  journal = {arXiv preprint arXiv:2602.17211},
  year    = {2026}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## Related Work

- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797) — Albergo et al., 2023
- [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456) — Song et al., 2021
- [Scale Dependencies and Self-Similar Models with Wavelet Scattering Spectra](https://arxiv.org/abs/2204.10177) — Morel, Rochette, Leonarduzzi, Bouchaud, Mallat (2022)
- [Scattering Spectra Models for Physics](https://arxiv.org/abs/2306.17210) — Cheng, Morel, Allys, Ménard, Mallat (2024)
