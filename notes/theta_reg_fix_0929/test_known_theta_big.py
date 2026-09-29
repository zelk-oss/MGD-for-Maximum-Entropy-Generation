exec(open(__file__.replace('_big', '')).read().split("e0 = score")[0])
for mode in ['legacy', 'fixed']:
    for lam in [1.0, 10.0, 100.0]:
        e = score(lam, mode)
        print(f'{mode:6s} lam={lam:6g}: rel RMS err {e[0]:.3f}   final node {e[1]:.3f}')
