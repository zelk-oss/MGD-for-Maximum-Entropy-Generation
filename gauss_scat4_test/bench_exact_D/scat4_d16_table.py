import pandas as pd
R = '/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation/gauss_scat4_test/runs/d16/results/'
h = pd.read_csv(R + 'H_gauss4.csv')
p = pd.read_csv(R + 'pred_H.csv').set_index('group')
pd.set_option('display.width', 200)
for g in ('fam:Scattering_Fourth_Order_Mod2_Real_Q1', 'fam:Scattering_Fourth_Order_Mod2_Imag_Q1'):
    d = h[h.group == g].copy()
    d['ESS'] = d.ess_frac * d.N
    d['pred'] = p.loc[g, 'H_pred']
    d['D_hat/pred'] = d.H / d.pred
    print(g.replace('Scattering_Fourth_Order_Mod2_', 'S4 '), '(', int(d.n_terms.iloc[0]), 'terms )')
    print(d[['seed', 'H', 'boot_std', 'pred', 'D_hat/pred', 'var2', 'ESS']].rename(columns={'H': 'D_hat', 'var2': 'half_Var'}).round(5).to_string(index=False))
    print('mean D_hat', round(d.H.mean(), 5), '| seed std', round(d.H.std(), 5), '| all seeds ESS >= 100:', bool((d.ESS >= 100).all()), '\n')
