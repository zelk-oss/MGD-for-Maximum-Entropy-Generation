"""Critically decimated, periodic, orthonormal Daubechies DWT (on top of Wavelet.dwt/idwt).

Channels follow the filter-bank convention of the potentials: channel l < levels is the
detail of level l+1 (0 = finest, d/2^(l+1) coefficients), channel `levels` is the
approximation (d/2^levels coefficients). Total = d, and W^T = W^-1 (machine precision with
the float64 taps). With d = 2^levels the approximation is the single coefficient sum(x)/sqrt(d).

    cd codes && python3 -m ortho_wavelet.periodic_dwt      # built-in tests
"""
import torch

from .Wavelet import Wavelet


class PeriodicDWT:
    def __init__(self, order, levels, d, device='cpu'):
        from filters.filters_1d import DAUBECHIES_H          # float64 taps, sum = sqrt(2)
        if d % 2 ** levels:
            raise ValueError(f"d = {d} is not divisible by 2^levels = {2 ** levels}")
        h = torch.tensor(DAUBECHIES_H[order], dtype=torch.float64)[None, None]
        self.order, self.levels, self.d = order, levels, d
        self.wavelet = Wavelet(h, 0, h.shape[-1] - 1, mode='Periodic', device=device).double()
        self.sizes = [d // 2 ** (l + 1) for l in range(levels)] + [d // 2 ** levels]
        self.n_channels = levels + 1
        self.n_max = max(self.sizes)
        self.mask = torch.zeros(self.n_channels, self.n_max, dtype=torch.bool, device=device)
        for j, n in enumerate(self.sizes):
            self.mask[j, :n] = True
        self.counts = torch.tensor(self.sizes, dtype=torch.float64, device=device)

    def to(self, device):
        self.wavelet = self.wavelet.to(device)
        self.mask, self.counts = self.mask.to(device), self.counts.to(device)
        return self

    def analysis(self, x):
        """x (B, 1, d) -> coefficients (B, n_channels, n_max), channel j zero-padded beyond sizes[j]."""
        a = x.reshape(-1, 1, self.d).double()
        out = a.new_zeros(a.shape[0], self.n_channels, self.n_max)
        for l in range(self.levels):
            a, det = self.wavelet.dwt(a)
            out[:, l, :det.shape[-1]] = det[:, 0]
        out[:, self.levels, :a.shape[-1]] = a[:, 0]
        return out.to(x.dtype)

    def synthesis(self, c):
        """Inverse (= adjoint) of analysis: (B, n_channels, n_max) -> (B, 1, d)."""
        a = c[:, self.levels:, :self.sizes[-1]].double()
        for l in reversed(range(self.levels)):
            a = self.wavelet.idwt(a, c[:, l:l + 1, :self.sizes[l]].double())
        return a.to(c.dtype)

    def synthesis_channel(self, cj, j):
        """Synthesis of channel j alone (all other channels zero): (N, sizes[j]) -> (N, d).
        Only the levels at and below j are inverted, O(d) per row."""
        a = cj.double()[:, None]
        if j < self.levels:
            a = self.wavelet.idwt(torch.zeros_like(a), a)
            below = j
        else:
            below = self.levels
        for _ in range(below):
            a = self.wavelet.idwt(a, torch.zeros_like(a))
        return a[:, 0].to(cj.dtype)


if __name__ == '__main__':
    d = 256
    for order in (4,):
        for levels in (1, 5, 8):
            W = PeriodicDWT(order, levels, d)
            E = torch.eye(d, dtype=torch.float64)[:, None]
            C = W.analysis(E)                                           # (d, C, n_max)
            A = torch.cat([C[:, j, :n] for j, n in enumerate(W.sizes)], -1).T   # (d, d): column i = W e_i
            I = torch.eye(d, dtype=torch.float64)
            rec = W.synthesis(C)[:, 0]
            per_ch = max((W.synthesis_channel(C[:, j, :n], j)
                          - W.synthesis(C * (torch.arange(W.n_channels)[None, :, None] == j))[:, 0]).abs().max().item()
                         for j, n in enumerate(W.sizes))
            print(f'db{order} levels={levels}: sizes {W.sizes} (sum {sum(W.sizes)}, expected d*2^-(l+1) '
                  f'{[d // 2 ** (l + 1) for l in range(levels)]} + {d // 2 ** levels})')
            print(f'   max|W^T W - I| {(A.T @ A - I).abs().max():.1e}   max|W W^T - I| {(A @ A.T - I).abs().max():.1e}   '
                  f'max|synthesis(analysis(x)) - x| {(rec - I).abs().max():.1e}   single-channel synthesis {per_ch:.1e}')
            if levels == 8:
                print(f'   approximation coefficient = sum(x)/sqrt(d): max err {(A[-1] - d ** -0.5).abs().max():.1e}')
