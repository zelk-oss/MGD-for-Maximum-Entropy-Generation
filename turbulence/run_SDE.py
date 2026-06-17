# run_sde.py

import argparse
import json
import torch
from pathlib import Path
import sys

# ── paths ─────────────────────────────────────────────
root = Path().resolve()
sys.path.insert(0, str(root / '../codes'))

# ── imports (unchanged) ───────────────────────────────
from sde_routines_condi import *
from sde_routines import *
from potentials_builder import *
from filters_bank import *
from utils import *
from utils_entropy import *
from potentials import *
from regularised_theta import *

sys.path.insert(0, str(root / '../data'))
from data_loader import *

# ── args ─────────────────────────────────────────────
parser = argparse.ArgumentParser()

# SDE core
parser.add_argument('--n1', type=int, required=True)
parser.add_argument('--nt', type=int, required=True)
parser.add_argument('--sigma', type=float, required=True)
parser.add_argument('--interpolant', type=str, default='Cos')
parser.add_argument('--schedule_exponent', type=float, default=1.0)

# scattering
parser.add_argument('--scales', type=int, required=True)
parser.add_argument('--J', type=int, required=True)
parser.add_argument('--Q', type=int, required=True)

# optimization
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--regularization', type=float, default=1e-1)

# potentials
parser.add_argument('--terms', nargs='+', required=True)

# misc
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--force_rerun', action='store_true')

args = parser.parse_args()

# ── device ───────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── load + preprocess data (kept identical) ──────────
W = DefineWavelet('Db', m=3, device=device)

Data = load_turbulence_1d()
Data = split_periodize_reshape(Data, args.n1)

scales = args.scales 
for _ in range(scales):
    Data = W.decompose(Data)[1]

x1 = normalize(Data[:args.n1]).to(device)

B, channels, M = x1.shape

# ── filters ──────────────────────────────────────────
filters, filters_Phi = return_Filters(M, args.J, 1, device=device, include_phi=True)
filters_Q = return_Filters(M, args.J, args.Q, device=device)

# ── time grid ────────────────────────────────────────
t = 1 - (1 - torch.linspace(0, 1, args.nt + 1))**args.schedule_exponent

# ── batch size default ───────────────────────────────
batch_size = args.batch_size
if batch_size is None:
    batch_size = args.n1

# ── config string (important for saving!) ─────────────
terms_str = "_".join(args.terms)

config = (
    f"scales{scales}_J{args.J}_Q{args.Q}"
    f"_n1{args.n1}_nt{args.nt}"
    f"_sigma{args.sigma}"
    f"_{args.interpolant}"
    f"_terms_{terms_str}"
)

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# ── save config (NEW FEATURE) ─────────────────────────
config_dict = vars(args)
config_dict["config_name"] = config

with open(outdir / f"{config}_config.json", "w") as f:
    json.dump(config_dict, f, indent=4)

# ── load or run ──────────────────────────────────────
def try_load():
    try:
        xt, theta_t, dH_t_bound, t_loaded = load_results(outdir, config)
        return xt, theta_t, dH_t_bound, t_loaded
    except:
        return None

if not args.force_rerun:
    loaded = try_load()
    if loaded is not None:
        print("Loaded existing results.")
        exit()

# ── potentials ───────────────────────────────────────
potentials = get_1d_potentials(
    args.terms,
    args.J,
    filters,
    args.Q,
    filters_Q,
    filters_Phi,
    scalar_param=None,
    parallel=False,
)

# ── solver ───────────────────────────────────────────
Solver = SDE(
    x1,
    x1.shape[0],
    x1.shape[0],
    t,
    args.sigma,
    potentials,
    batch_size,
    device=device,
    regularization=args.regularization,
    interpolant=args.interpolant,
)

xt, barphi_e, barphi_p, eta_t, theta_t, dH_t_bound = Solver()

# ── save (unchanged structure!) ──────────────────────
save_results(xt, theta_t, dH_t_bound, t, outdir, config)

torch.save(
    {
        'barphi_e': barphi_e.detach().cpu(),
        'barphi_p': barphi_p.detach().cpu(),
    },
    outdir / f"{config}_aux.pt",
)

print("Finished:", config)