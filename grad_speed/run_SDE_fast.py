"""turbulence/run_SDE.py with the fast eager potentials (2026-11-23, notes/fast_families_report_1122.md).

Same command line as turbulence/run_SDE.py. Right before the SDE loop starts, every potential's grad / forward is replaced
by the new eager version (Scat4 J/F/I/A1, Scat3 J/A/V, GGD R); L_6 keeps its original grad(x, v). Nothing in codes/ is
changed. Used through the launcher with PYTHON_SCRIPT=grad_speed/run_SDE_fast.py.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'grad_speed'):
    sys.path.insert(0, str(p))
from codes.sde_routines import SDE                                              # noqa: E402
from validate_families_fast import patch                                        # noqa: E402

_forward_regularised = SDE.forward_regularised


def forward_regularised(self, *args, **kwargs):
    patch(self, None, 'new')
    print('run_SDE_fast: fast eager potentials for ' + ', '.join(self.potentials), flush=True)
    return _forward_regularised(self, *args, **kwargs)


SDE.forward_regularised = forward_regularised
script = ROOT / 'turbulence' / 'run_SDE.py'
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name='__main__')
