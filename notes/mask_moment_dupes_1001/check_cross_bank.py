"""Cross-bank near-copies (2026-10-01): Morlet bank (Q=1) vs deduplicated psi bank (Q=3).

Prints, for M=256, J=8, dedup tol 1e-8 (the zfloor runs):
  (a) the filter deficit 1 - |<f,g>|/(|f||g|) of each Q=1 filter to its closest kept psi filter;
  (b) 1 - |corr| of the L_6 gradients (L2p_norm(3, .)) of the pairs (7,20), (8,21) on
      N synthetic signals (white, and spectrum k^-2), pooled over signals and positions.
Usage (project root): python notes/mask_moment_dupes_1001/check_cross_bank.py
"""
import sys
sys.path[:0] = ['.', 'codes', 'data']
import torch
from codes.filters_bank import return_Filters, deduplicate_filters
from codes.potentials.potentials_1d import L2p_norm

M, J, N = 256, 8, 4000
F1 = return_Filters(M, J, 1)
F3d, kept = deduplicate_filters(return_Filters(M, J, 3), tol=1e-8)
print('kept psi channels:', kept)

A = F1.reshape(-1, M).to(torch.complex128); B = F3d.reshape(-1, M).to(torch.complex128)
C = (A.conj() @ B.T).abs() / (A.abs().pow(2).sum(-1).sqrt()[:, None] * B.abs().pow(2).sum(-1).sqrt()[None, :])
D = 1 - C
for i in range(A.shape[0]):
    j = int(D[i].argmin())
    print(f'(a) Q=1 filter {i}: closest kept psi {j} (bank channel {kept[j]}), deficit {float(D[i, j]):.2e}')

torch.manual_seed(0)
P1, P3 = L2p_norm(3, F1), L2p_norm(3, F3d)
w = (torch.fft.fftfreq(M) * M).abs()
for name, amp in [('white', torch.ones(M)), ('spectrum k^-2', 1 / torch.clamp(w, min=1))]:
    x = torch.fft.ifft(torch.fft.fft(torch.randn(N, M, dtype=torch.float64)) * amp).real
    x = (x / x.std()).float()[:, None, :]
    g1, g3 = P1.grad(x).double(), P3.grad(x).double()
    for a, b in [(7, 20), (8, 21)]:
        u, v = g1[:, a], g3[:, b]
        c = (u * v).sum() / ((u * u).sum().sqrt() * (v * v).sum().sqrt())
        print(f'(b) {name}: L_6[{a}] vs L_6_psi[{b}]: 1-|corr| = {1 - abs(float(c)):.2e}')
