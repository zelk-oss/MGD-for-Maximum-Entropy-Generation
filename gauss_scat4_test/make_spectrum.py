#!/usr/bin/env python3
"""Spectrum of a stationary Gaussian that lies EXACTLY in the order-2 scattering family (Scattering_Second_Order_Q1,
Morlet Q = 1, M, J): the max-ent Gaussian p* with the order-2 moments of a source spectrum (white, or a saved
gauss_spectrum.pt such as the Lagrangian one). Data drawn from S* make the max-ent solution of ANY model containing
Scattering_Second_Order_Q1 equal to p* itself, with theta = 0 on every other term (L_6, order 4): uniqueness of max-ent.

Conventions: S = unnormalised power spectrum E|FFT x|^2 (codes/gauss_spectrum.py: sample_gaussian_spectrum, true_log_p);
C = S / M (ortho FFT) = eigenvalues of the circulant covariance. S* is rescaled to unit variance per coordinate (the
family is invariant under x -> c x: theta -> theta / c^2). Exact entropy H(p*) = 1/2 sum_w log(2 pi e C*_w) (nats).
theta_star = Scattering_Second_Order_Q1 coefficients of p* in the code sign (log p = +theta . phi + const).

Usage: python gauss_scat4_test/make_spectrum.py --M 16 --J 4 --source white --out <file.pt>
       python gauss_scat4_test/make_spectrum.py --M 256 --J 8 --source <exp_dir>/gauss_spectrum.pt --out <file.pt>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', ROOT / 'turbulence'):
    sys.path.insert(0, str(p))
from filters_bank import return_Filters                                      # noqa: E402
from logz_gauss_benchmark import max_ent_from_spectrum                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--M', type=int, required=True)
ap.add_argument('--J', type=int, required=True)
ap.add_argument('--source', required=True, help="'white' or a gauss_spectrum.pt with S (1, M), unnormalised")
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()

M = a.M
psi2 = (return_Filters(M, a.J, 1)[0].abs() ** 2).double().numpy()            # (J + 1, M): |psihat_l(w)|^2 incl. low-pass
if a.source == 'white':
    C_src = np.ones(M)
else:
    S_src = torch.load(a.source, map_location='cpu', weights_only=False)['S'].double().numpy().reshape(-1)
    assert S_src.size == M, (S_src.shape, M)
    C_src = S_src / M
C_src = C_src * M / C_src.sum()                                               # unit variance per coordinate
star = max_ent_from_spectrum(C_src, psi2, M)
assert np.isfinite(star['H_star']), f'max-ent did not converge: {star["moment_rel_err"]:.2e}'
C = np.asarray(star['spectrum_star'], dtype=np.float64)
H_src = 0.5 * np.log(2 * np.pi * np.e * C_src).sum()
# KL(source || p*) = H(p*) - H(source) holds BEFORE rescaling (p* has the source's order-2 moments; max-ent identity)
KL_src = 0.5 * np.log(2 * np.pi * np.e * C).sum() - H_src
c = M / C.sum()                                                               # then rescale p* to unit variance
C = C * c
H_star = 0.5 * np.log(2 * np.pi * np.e * C).sum()
# theta_star from lam(w) = -theta . A(w) = 1 / C(w)  (max_ent_from_spectrum algebra): least squares, exact in the family
A = (psi2 + psi2[:, (-np.arange(M)) % M]) / M
theta_star, res, *_ = np.linalg.lstsq(A.T, -1.0 / C, rcond=None)
fit = np.abs(theta_star @ A + 1.0 / C).max() * C.max()
torch.save(dict(S=torch.as_tensor(C * M)[None], theta_star=torch.as_tensor(theta_star), H_star=float(H_star),
                H_source=float(H_src), KL_source_to_star=float(KL_src), rescale=float(c), M=M, J=a.J, source=a.source,
                psi2=torch.as_tensor(psi2)), a.out)
print(f'M {M} J {a.J} source {a.source}: max-ent moment err {star["moment_rel_err"]:.1e} | H(p*) {H_star:.4f} nats | '
      f'H(source) {H_src:.4f} | KL(source||p*) {KL_src:.4f} (before rescaling by {c:.4f}) | in-family fit rel err {fit:.1e} | -> {a.out}')
