"""Summarise a cProfile dump of run_SDE.py (see profile_step.sh).

Prints the SDE-step functions sorted by cumulative time, with their share of the
step loop, and splits compute_G / compute_moments by caller, so the cost of the
duplicated Gram at y_k (called from forward_regularised) is read off directly.

Usage: python launch/read_profile.py <file.prof>
"""
import pstats
import sys

FUNCS = ['forward_regularised', 'iteration_step_projection', 'compute_eta', 'compute_theta',
         'compute_G', 'compute_moments', 'compute_grad_phi_projected', 'compute_grad_potentials',
         'compute_interpolant', 'compute_rhs_dt_phi_I_t', 'compute_rhs_constraint_correction',
         'solve', '_solve_regularised_thomas']
SPLIT = ['compute_G', 'compute_moments']


def main(path):
    st = pstats.Stats(path)
    rows = {}                                   # name -> (ncalls, cumtime), summed over files
    for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
        if name in FUNCS and ('sde_routines' in fn or name == 'solve'):
            n, c = rows.get(name, (0, 0.0))
            rows[name] = (n + nc, c + ct)
    steps = rows.get('iteration_step_projection', (1, 0))[0]
    # cProfile can under-count forward_regularised (tqdm's monitor thread takes over
    # the caller frames: work done directly in the loop shows up under 'wait'), so
    # the loop time is taken as the larger of it and the per-step work.
    loop = max(rows.get('forward_regularised', (1, 0.0))[1],
               rows.get('iteration_step_projection', (1, 0.0))[1])
    print(f'forward_regularised: {loop:.1f} s total, {steps} steps -> {loop / max(steps, 1):.3f} s/step '
          f'(blocking launches: slower than a normal run)\n')
    print(f'{"function":36s} {"calls":>7s} {"calls/step":>10s} {"s/call":>8s} {"cum s":>9s} {"% of loop":>9s}')
    for name, (n, c) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        print(f'{name:36s} {n:7d} {n / max(steps, 1):10.2f} {c / max(n, 1):8.3f} {c:9.1f} '
              f'{100 * c / loop:8.1f}%')

    for target in SPLIT:
        print(f'\n{target} by caller:')
        for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
            if name != target or 'sde_routines' not in fn:
                continue
            for (cfn, cline, cname), val in sorted(callers.items(), key=lambda kv: -kv[1][3]):
                # val = (primitive calls, total calls, tottime, cumtime) for this caller
                print(f'   {cname:34s} line {cline:5d}  calls {val[1]:6d}  cum {val[3]:8.1f} s  '
                      f'({100 * val[3] / loop:5.1f}% of loop)')


if __name__ == '__main__':
    main(sys.argv[1])
