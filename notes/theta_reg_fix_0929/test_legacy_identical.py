# Legacy mode of the new _solve_regularised_thomas must be bit-identical to HEAD's solver.
import os as _os
REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
import sys, types, ast, numpy as np, torch
sys.path[:0] = [REPO,
                REPO + '/codes',
                REPO + '/data']
from codes.sde_routines import SDE
# HEAD's function, extracted from the file without importing the module
src = open(sys.argv[1]).read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SDE')
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_solve_regularised_thomas')
ns = {'np': np, 'torch': torch}
exec(compile(ast.Module([fn], []), 'head', 'exec'), ns)
old = ns['_solve_regularised_thomas']

rng = np.random.default_rng(1)
n, r = 60, 7
t = np.sort(rng.uniform(0.01, 0.999, n))
def spd():
    A = rng.standard_normal((r, r)); return torch.tensor(A @ A.T + 1e-3 * np.eye(r), dtype=torch.float32)
M = [spd() for _ in range(n)]; G = [spd() for _ in range(n)]
b = [torch.tensor(rng.standard_normal(r), dtype=torch.float32) for _ in range(n)]
c = [torch.tensor(rng.standard_normal(r), dtype=torch.float32) for _ in range(n)]
for lam in [0.0, 1e-6, 1e-3, 1.0]:
    for ridge in [0.0, 1e-4]:
        s1 = types.SimpleNamespace(num_potentials=r); s2 = types.SimpleNamespace(num_potentials=r)
        A = old(s1, t, M, G, b, c, lam, ridge=ridge)
        B = SDE._solve_regularised_thomas(s2, t, M, G, b, c, lam, ridge=ridge)
        print(f'lam={lam:g} ridge={ridge:g}: bit-identical={torch.equal(A, B)}, '
              f'residuals {s1.last_reg_residual:.1e} / {s2.last_reg_residual:.1e}')
