"""log p / entropy / concentration of finished nested-model runs -> nested_models/results/<label>/seed<k>.npz

Run on Jean Zay (needs saved_results/ and the offline Theta_reg of each run, resolve_lamsweep.sh), from the
repo root, on a GPU (dev QoS is enough). Only the small .npz files need to be pulled afterwards.

    python nested_models/analyse_nested.py                       # every model of models.py, seeds 900-904
    python nested_models/analyse_nested.py --models "M2 Morlet" --seeds 900 901
    python nested_models/analyse_nested.py --labels turbulence_nested_M2 --seeds 900

A (model, seed) whose run or Theta_reg is not there yet is reported and skipped, so the same command can be
repeated while the runs finish; existing .npz are kept unless --force.
"""
import argparse
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logp_entropy as le                                           # noqa: E402
from models import MODELS, SEEDS                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None, help='display names of models.py (default: all)')
    ap.add_argument('--labels', nargs='*', default=None, help='raw --label values (overrides --models)')
    ap.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    ap.add_argument('--lam', type=float, default=5e-4)
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--n_model', type=int, default=None, help='use only the first N final samples')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    labels = a.labels or [MODELS[m] for m in (a.models or MODELS)]
    for label in labels:
        for seed in a.seeds:
            f = HERE / 'results' / label / f'seed{seed}.npz'
            if f.exists() and not a.force:
                print(f'[keep] {label} seed {seed}: {f.name} exists')
                continue
            try:
                out = le.analyse(label, seed, a.lam, a.ridge, n_model=a.n_model)
            except (FileNotFoundError, KeyError) as e:
                print(f'[skip] {label} seed {seed}: {e}')
                continue
            except Exception:
                print(f'[FAIL] {label} seed {seed}')
                traceback.print_exc()
                continue
            le.save(out)
            print(f"[done] {label} seed {seed}: H_bound {out['H_bound']:.2f}  H_sample {out['H_sample']:.2f}  "
                  f"gap {out['gap']:+.2f}  std(-log p) model {out['std_model']:.2f}  data {out['std_data']:.1f}  "
                  f"-> {f}")


if __name__ == '__main__':
    main()
