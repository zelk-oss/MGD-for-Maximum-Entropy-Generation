"""Gradient check of the 2D potentials against autograd (tiny sizes, CPU, a few seconds).

For each potential: grad(x) (all statistics) and grad(x, v) (contracted with a random v) are compared with the
Jacobian of forward() obtained by torch.autograd. Run:  python3 cosmic_web/test_potentials_2d.py
The Real classes are included as a control: they are the existing, validated ones.
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
for p in (project_root, project_root / 'codes', project_root / 'data'):
    sys.path.insert(0, str(p))

from codes.filters_bank import return_Filters                      # noqa: E402
from codes.potentials_builder import get_2d_potentials             # noqa: E402

torch.manual_seed(0)
M, J, L, B = 16, 2, 2, 2
filters = return_Filters(M, J, 1, L=L, device='cpu')
print('filters', tuple(filters.shape), 'complex:', filters.is_complex())

terms = ['L_6', 'Scattering_First_Order', 'Scattering_Second_Order',
         'Scattering_Third_Order_Real', 'Scattering_Third_Order_Imag',
         'Scattering_Fourth_Order_Real', 'Scattering_Fourth_Order_Imag',
         'Scattering_Fourth_Order_Mod2_Real', 'Scattering_Fourth_Order_Mod2_Imag']
pots = get_2d_potentials(terms, J, L, filters)
x = torch.randn(B, 1, M, M)

worst = 0.0
for name, pot in pots.items():
    r = pot.num_coefficients
    y = pot.forward(x)
    assert y.shape == (B, r), (name, y.shape, r)
    # Jacobian per sample, (r, M, M)
    jac = torch.stack([torch.autograd.functional.jacobian(lambda z: pot.forward(z[None])[0], x[b]).reshape(r, M, M)
                       for b in range(B)])
    g = pot.grad(x).reshape(B, r, M, M)
    err_all = ((g - jac).abs().max() / jac.abs().max()).item()
    v = torch.randn(r)
    gv = pot.grad(x, v=v).reshape(B, 1, M, M)
    jv = torch.einsum('r,brmn->bmn', v, jac)[:, None]
    err_v = ((gv - jv).abs().max() / jv.abs().max()).item()
    worst = max(worst, err_all, err_v)
    print(f'{name:32s} r={r:4d}  grad(x) rel err {err_all:.2e}   grad(x,v) rel err {err_v:.2e}')
print('worst relative error:', f'{worst:.2e}')
