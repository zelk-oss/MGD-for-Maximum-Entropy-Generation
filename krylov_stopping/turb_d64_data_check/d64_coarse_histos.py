"""Wavelet-coefficient histograms (run_SDE's own plot) for the current data and the d = 64 spread data."""
import importlib.util, os, sys
import numpy as np, torch
ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location('run_SDE', f'{ROOT}/turbulence/run_SDE.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
OUT = sys.argv[1]

raw = R.load_turbulence_1d().to(R.device)
W = R.DefineWavelet('Db', m=3, device=R.device)


def build(sub, target):
    x = R.split_periodize_reshape(raw, sub)
    for _ in range(int(np.log2(x.shape[-1] / target))):
        x = W.decompose(x)[1]
    return x


cur = None
dt = build(512, 64)                                    # coarse x3, same selection as run_SDE --n1_spread
per = dt.shape[0] // raw.shape[0]
g = np.random.default_rng(0)
series = g.choice(raw.shape[0], 2000, replace=False)
new = R.normalize(dt[torch.as_tensor(series * per + g.integers(0, per, 2000))])
print('new', tuple(new.shape), '| distinct trajectories in new:', len(set(series)))

for name, x, J in (('coarse3_d64_J6', new, 6),):
    for Q in (1, 3):
        f = f'{OUT}/histo_{name}_Q{Q}.png'
        R.plot_wavelet_coeff_histograms(x, J, Q, name, None, None, save={'filename': f, 'title': f'{name}, Q={Q}'})
        print('wrote', f)
