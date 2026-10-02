import sys, numpy as np, torch, glob
from scipy import stats
S = sys.argv[1] if len(sys.argv) > 1 else '.'   # folder with phi_sep.pt (logp_decompose.py --save)
phi = torch.load(f'{S}/phi_sep.pt').numpy()
files = sorted(glob.glob('saved_results/theta_last/*.pt'))
TH = np.stack([torch.load(f).double().numpy() for f in files])           # (5, 281)
seeds = [f.split('_seed_')[1][:3] for f in files]
names = {7: 'L_6[7]', 29: 'L_6_psi[20]', 27: 'L_6_psi[18]', 28: 'L_6_psi[19]', 109: 'Scalar_psi[77]',
         140: 'Scalar_morlet[30]', 151: 'Scat4_Real[10]', 152: 'Scat4_Real[11]', 153: 'Scat4_Real[12]',
         144: 'Scat4_Real[3]', 146: 'Scat4_Real[5]', 147: 'Scat4_Real[6]', 224: 'Scat4_Real[83]', 8: 'L_6[8]'}
print('theta per seed on the columns behind the outliers  (seeds ' + ' '.join(seeds) + ')')
for c, n in names.items():
    v = TH[:, c]
    print(f'  col {c:3d} {n:18s} ' + ' '.join(f'{x: .3g}' for x in v) +
          f'   | mean {v.mean(): .3g}  std/|mean| {v.std(ddof=1) / max(abs(v.mean()), 1e-300):.2f}')
print('\nlog p per seed (each seed its own theta):')
LP = phi @ TH.T                                                          # (n1, 5)
for s, lp in zip(seeds, LP.T):
    print(f'  seed {s}: std {lp.std():9.4g}  skew {stats.skew(lp):7.3g}  excess kurt {stats.kurtosis(lp):8.3g}  '
          f'min-med {lp.min() - np.median(lp):+9.4g}  max-med {lp.max() - np.median(lp):+9.4g}')
C = np.corrcoef(LP.T)
print('  corr of log p across seeds (signal by signal):\n' + '\n'.join('   ' + ' '.join(f'{x:6.3f}' for x in row) for row in C))
# the pair sum and difference: only theta_a + theta_b * (phi_b/phi_a scale) is fixed by moments
for a, b in [(109, 140), (7, 29)]:
    print(f'\n  {names[a]} + {names[b]} per seed: theta sum ' + ' '.join(f'{x: .3g}' for x in TH[:, a] + TH[:, b]))
