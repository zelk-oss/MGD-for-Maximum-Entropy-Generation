"""
Moment-Guided Diffusion (MGD) - utilities.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, Any, Tuple
import sys 

_UTILS_DIR = Path(__file__).resolve().parent
_UTILS_PROJECT_ROOT = _UTILS_DIR.parent
 
def _extend_syspath(root: Path, project_root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    codes_path = project_root / 'codes'
    if codes_path.is_dir() and str(codes_path) not in sys.path:
        sys.path.insert(0, str(codes_path))
    data_path = project_root / 'data'
    if data_path.is_dir() and str(data_path) not in sys.path:
        sys.path.insert(0, str(data_path))

_extend_syspath(_UTILS_DIR, _UTILS_PROJECT_ROOT)


from codes.check_moments import *     # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import *

import math
import pandas as pd


class TensorDataset(torch.utils.data.Dataset):
    """ We have to create our own class because PyTorch's TensorDataset returns lists... """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]

def add_noise(x, t):
    """Apply Ornstein-Uhlenbeck noising at time ``t``.
 
    Returns ``x_t = e^{-t} x + sqrt(1 - e^{-2t}) z`` with ``z`` standard Gaussian.
 
    Parameters
    ----------
    x : torch.Tensor
        Clean input.
    t : float
        Noise time.
 
    Returns
    -------
    x_t : torch.Tensor
        Noised input.
    z : torch.Tensor
        The Gaussian noise used.
    """

    e_minus_t = np.exp(-t)
    std = np.sqrt(1 - e_minus_t ** 2)

    #torch.manual_seed(13)
    z = torch.randn_like(x)
    x_t = e_minus_t * x + std * z

    return x_t, z


def symmetrize_functional(x):
    """
    Functional version (no class needed).
    
    Args:
        x: Tensor of shape (B, C, M, N)
    Returns:
        Symmetrized tensor of shape (B, C, M*2, N*2)
    """

    # Create the 4 quadrants
    top = torch.cat([x, torch.flip(x, dims=[3])], dim=3)
    bottom = torch.cat([torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])], dim=3)
    return torch.cat([top, bottom], dim=2)

def save_results(
    xt,
    theta_t,
    dH_t_bound,
    t,
    root,
    config,
):
    base = root / 'saved_results'

    torch.save(xt.cpu(), base / 'samples' / config)
    torch.save(theta_t.cpu(), base / 'lagrange_multipliers' / config)
    torch.save(dH_t_bound.cpu(), base / 'entropy_bounds' / config)
    torch.save(t.cpu(), base / 'sampling_times' / config)

def save_results_theta_reg(
    xt,
    theta_t,
    dH_t_bound,
    t,
    root,
    config,
    Theta_reg=None,  # Set to None by default
):
    base = root / 'saved_results'

    # 1. Save the core variables first to secure them on disk
    torch.save(xt.cpu(), base / 'samples' / config)
    torch.save(theta_t.cpu(), base / 'lagrange_multipliers' / config)
    torch.save(dH_t_bound.cpu(), base / 'entropy_bounds' / config)
    torch.save(t.cpu(), base / 'sampling_times' / config)

    # 2. Safely process Theta_reg only if it was passed in
    if Theta_reg is not None:
        try:
            torch.save(
                Theta_reg.cpu(),
                base / 'lagrange_multipliers_regularised' / config
            )
        except RuntimeError as e:
            # Catch the OOM (or any other PyTorch error) so the run doesn't crash entirely
            print(f"Warning: Failed to save Theta_reg. Error: {e}")

def load_results(root: Path, exact_config: str) -> Tuple[Any, Any, Any, Any, Any]:
    base = root / 'saved_results'

    x_t = torch.load(base / 'samples' / exact_config)
    theta_t = torch.load(base / 'lagrange_multipliers' / exact_config)
    dH_t_bound = torch.load(base / 'entropy_bounds' / exact_config)
    t = torch.load(base / 'sampling_times' / exact_config)

    path_theta_reg = base / 'lagrange_multipliers_regularised' / exact_config

    if path_theta_reg.exists():
        Theta_reg = torch.load(path_theta_reg)
    else: 
        Theta_reg = None # Always return 5 items to prevent unpacking crashes
        
    return (x_t, theta_t, dH_t_bound, t, Theta_reg)

def normalize(Data):
    """Standardize ``Data`` to zero mean and unit std, cast to float32.
 
    Parameters
    ----------
    Data : torch.Tensor
        Input tensor.
 
    Returns
    -------
    torch.Tensor
        Normalized float32 tensor.
    """

    Data = (Data-Data.mean())/Data.std()
    Data = Data.to(torch.float32)
    return Data


def split_periodize_reshape(Data, n1):
    """
    Splits Data into non-overlapping subseries of length n1 along the last axis,
    discards the end if not divisible, periodizes each subseries, and reshapes
    the output to [batch * num_subseries, channels, n1].
    """
    num_subseries = Data.size(-1) // n1
    Data_sub = Data[:, :, :num_subseries * n1]
    Data_sub = Data_sub.unfold(-1, n1, n1)

    first = Data_sub[..., 0:1]
    last = Data_sub[..., -1:]
    a = (last - first) / (n1 - 1)
    b = first
    x = torch.arange(n1, device=Data.device, dtype=Data.dtype)
    linear = a * x + b 
    Data_periodized = Data_sub - linear 
    Data_periodized = Data_periodized + first 

    batch, channels, _, _ = Data_periodized.shape
    return Data_periodized.reshape(batch * num_subseries, channels, n1)



"""
theta_analysis_v2.py

Cleaned-up version of theta_analysis.py / theta_plotting.py, with four changes
relative to the original notebook code:

1. All the internal diagnostic prints (column maps, index-decoding fingerprint
   messages, etc.) are OFF by default. Every function still accepts
   verbose=True if you want to debug a specific case.

2. Legends are drawn on each subplot (ax.legend(...)) instead of a single
   fig.legend(...) floating above/outside the figure.

3. `Scattering_Fourth_Order_Mod2_Real_Q1` and `..._Imag_Q1` are no longer
   treated as two independent real potentials. `build_complex_theta_dataframe`
   pairs them coefficient-by-coefficient into theta_complex = real + i*imag,
   and the modulus used everywhere downstream (|theta|, variance, plots) is
   the modulus of that complex number, not of the real part alone.

4. `build_all_averaged_vectors` produces one readable dict {potential_name:
   averaged_theta_vector} per experiment type (MGD / Reg), averaged over
   seeds, with the Real/Imag Q1 pair collapsed into a single complex vector.
   These are plain 1D numpy arrays (real or complex) you can slice, plot, or
   feed into further analysis directly.

Assumptions worth flagging explicitly (adjust if wrong for your setup):
- Real/Imag potentials of the same family are assumed to be built with the
  same J, Q, offset, lite settings, so that every (j, s1, s2) coefficient the
  imaginary branch has is ALSO present in the real branch (just possibly not
  vice versa -- e.g. the real branch commonly keeps diagonal terms the
  imaginary branch prunes because they're identically zero). theta_complex is
  built on the real branch's layout; any real-branch coefficient absent from
  the imaginary branch gets an imaginary part of exactly 0.0. If the
  imaginary branch ever has a coefficient the real branch lacks,
  `expand_imag_to_real_indices` raises rather than silently dropping it.
- The moment-normalization for the complex case is
  theta_complex * (mk_real + i*mk_imag), i.e. the same elementwise
  fitted-times-moment convention as the real case, just extended to complex.
  This is a direct generalization, not something derived from first
  principles for the Mod2 potential -- double check it matches your intended
  physical normalization if you use `theta_complex_normalized` downstream.
"""

import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def expand_imag_to_real_indices(pot_im, pot_re, theta_im, verbose=False, index_convention='mod2'):
    """
    Re-express theta_im (defined on pot_im's own coefficient ordering) onto
    the coefficient layout of pot_re. Any index triplet that pot_re has but
    pot_im does not -- e.g. the diagonal terms, which are identically zero
    for an Im(z * conj(z))-type potential and so are typically pruned out
    of the imaginary branch entirely -- is filled with 0.0.

    Matching is done on the raw index triplet from `.indices` (3, n_coeff):
    one shared/singleton axis plus an unordered pair of the other two, so
    swapping the pair's order is treated as the same coefficient.

    index_convention: WHICH raw column is the singleton vs the pair matters
    -- getting it wrong doesn't just mislabel columns, it can silently
    CROSS-MATCH two physically different coefficients whose raw triples
    happen to collide under the wrong symmetry grouping (e.g. a genuine
    real 'diagonal' entry, which should get imaginary part exactly 0,
    picking up an unrelated surviving imaginary coefficient's value
    instead). Both 'mod2' and 'jab' now map to the SAME grouping -- pair =
    (col[0], col[1]), singleton = col[2] -- verified against every
    Scattering_Fourth_Order_{Real,Imag,Mod2_Real,Mod2_Imag}_1d potential's
    actual forward()/fit_micro tensor shapes (col[0],col[1] are the fine,
    first-layer s1,s2; col[2] is the coarse, second-layer j) -- see
    _decode_scattering_jsk. 'jab' used to assume a different, WRONG
    grouping (pair = (col[1], col[2]), singleton = col[0]) matching
    decode_jab's old (also wrong) index model; it's kept only as a
    backward-compatible alias for callers that still pass it explicitly.

    Parameters
    ----------
    pot_im, pot_re : potentials with a `.indices` tensor of shape (3, dim)
    theta_im : array-like, shape (dim_im,) or (T, dim_im). theta_im.shape[-1]
        must equal pot_im.indices.shape[-1] -- pass pot_im's OWN theta slice,
        not something already sliced to dim_re.

    Returns
    -------
    theta_im_full : ndarray, shape (dim_re,) or (T, dim_re), dtype matching
        theta_im. Position i holds pot_im's value whenever pot_re's i-th
        index triplet also appears in pot_im, and 0.0 otherwise.
    """
    idx_re = pot_re.indices.long().detach().cpu().numpy()   # (3, dim_re)
    idx_im = pot_im.indices.long().detach().cpu().numpy()   # (3, dim_im)
    dim_re = idx_re.shape[-1]
    dim_im = idx_im.shape[-1]

    theta_im_arr = theta_im.detach().cpu().numpy() if torch.is_tensor(theta_im) else np.asarray(theta_im)
    if theta_im_arr.shape[-1] != dim_im:
        raise ValueError(
            f"theta_im last dim ({theta_im_arr.shape[-1]}) doesn't match "
            f"pot_im.indices ({dim_im}) -- pass pot_im's own theta slice, "
            f"not one already sliced/expanded to the real potential's size."
        )

    if index_convention in ('mod2', 'jab'):
        def _key(col):
            s1, s2, shared = int(col[0]), int(col[1]), int(col[2])
            lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
            return (shared, lo, hi)
    else:
        raise ValueError(f"index_convention must be 'mod2' or 'jab', got '{index_convention}'")

    imag_map = {}
    for i in range(dim_im):
        key = _key(idx_im[:, i])
        if key in imag_map:
            raise ValueError(f"duplicate index {key} within the imaginary potential -- "
                              f"cannot build a unique coefficient mapping.")
        imag_map[key] = i

    is_2d = theta_im_arr.ndim == 2
    shape = (theta_im_arr.shape[0], dim_re) if is_2d else (dim_re,)
    theta_im_full = np.zeros(shape, dtype=theta_im_arr.dtype)

    n_matched = 0
    for i in range(dim_re):
        j = imag_map.get(_key(idx_re[:, i]))
        if j is not None:
            n_matched += 1
            theta_im_full[..., i] = theta_im_arr[..., j]
        # else: coefficient exists on the real branch but not the imaginary
        # one (e.g. diagonal) -- leave at 0.0.

    if verbose:
        print(f"expand_imag_to_real_indices: matched {n_matched}/{dim_re} real coefficients "
              f"({dim_re - n_matched} filled with 0.0); imaginary potential had {dim_im} coefficients.")
    if n_matched < dim_im:
        raise ValueError(
            f"{dim_im - n_matched} imaginary coefficients did not match any real-potential "
            f"index -- the imaginary branch has entries the real branch doesn't, so it can't "
            f"be losslessly expanded onto the real layout. Check J/Q/offset/lite settings match."
        )

    return theta_im_full


# ---------------------------------------------------------------------
# 1. Column mapping: potentials dict -> theta_t column slices
# ---------------------------------------------------------------------
def _potential_width(p, key, known_dims=None):
    """(dim, source) for potential p. Never silently assumes dim=1."""
    known_dims = known_dims or {}
    for attr in ('n_theta', 'num_theta', 'theta_dim'):
        if hasattr(p, attr):
            val = getattr(p, attr)
            val = val() if callable(val) else val
            if torch.is_tensor(val):
                val = val.numel()
            return int(val), f"'{attr}'"
    if hasattr(p, 'num_coefficients'):
        return int(p.num_coefficients), "'num_coefficients'"
    if hasattr(p, 'indices'):
        idx = p.indices
        width = idx.shape[-1] if torch.is_tensor(idx) else np.asarray(idx).shape[-1]
        return int(width), "'.indices.shape[-1]'"
    for attr in ('num_regions', 'n_regions'):
        if hasattr(p, attr):
            n_reg = int(getattr(p, attr))
            n_par = int(getattr(p, 'n_params_per_region', 1))
            return n_reg * n_par, f"'{attr}' x n_params_per_region"
    if key in known_dims:
        return int(known_dims[key]), "known_dims override (UNVERIFIED by introspection)"
    raise RuntimeError(f"Cannot determine theta width for '{key}' ({type(p).__name__}).")


def theta_column_map(potentials, n_theta, known_dims=None, verbose=False):
    """
    dict: key -> (start, dim) column slice into theta_t, for every potential
    in `potentials` (dict, insertion-order must equal theta_t column order,
    which is how get_1d_potentials + run_experiment build it).
    """
    keys = list(potentials.keys())
    dims, sources = [], []
    for k in keys:
        d, src = _potential_width(potentials[k], k, known_dims)
        dims.append(d)
        sources.append(src)
    offsets = np.cumsum([0] + dims)
    if sum(dims) != n_theta:
        lines = "\n".join(f"  {k:35s} {d:5d}  {s}" for k, d, s in zip(keys, dims, sources))
        raise AssertionError(
            f"sum(dims)={sum(dims)} != theta_t.shape[1]={n_theta}\n{lines}\n\n"
            f"Likely cause: `potentials` here is the freshly-constructed, NOMINAL "
            f"potentials dict (e.g. straight from get_1d_potentials(...)), whose "
            f"coefficient counts are the theoretical/un-pruned sizes. Per-seed "
            f"fitting can nondeterministically prune potentials (e.g. "
            f"Scalar_GGD_KRegion-based ones) down to fewer coefficients. Pass the "
            f"ACTUAL fitted potentials for a seed whose layout matches this theta "
            f"(e.g. via get_potentials_for_seed(...) / load_potentials_for_config(...)), "
            f"not the plain get_1d_potentials(...) output. See "
            f"find_matching_reference_potentials() for picking one automatically."
        )
    if verbose:
        for k, d, o, s in zip(keys, dims, offsets[:-1], sources):
            print(f"{k:35s} {d:5d}  [{o:4d}:{o+d:<4d}]  {s}")
    return {k: (int(o), int(d)) for k, o, d in zip(keys, offsets[:-1], dims)}


def find_matching_reference_potentials(theta_all, potentials_by_seed, known_dims=None):
    """
    `theta_all` (n_seeds, n_features) was built by stacking each seed's
    theta -- which only works if every kept seed happens to share the same
    pruned column layout. This picks out ONE such seed's actual FITTED
    potentials dict (not the nominal, un-pruned one) so you have something
    valid to pass to theta_column_map / build_all_averaged_vectors.

    potentials_by_seed: dict {seed_key: potentials_dict}, where each
    potentials_dict is the per-seed FITTED potentials, e.g. as returned by
    your own get_potentials_for_seed(...) / load_potentials_for_config(...)
    -- collect these into a dict as you loop over seeds building dfs/dfs_reg,
    the same way `potentials_by_seed[key] = potentials_k` was already being
    done in your seed-loading loop.

    Returns (seed_key, potentials_dict, col_map) for the first seed whose
    total coefficient count equals theta_all.shape[-1]. Raises with a
    per-seed breakdown if none match.
    """
    theta_np = theta_all.detach().cpu().numpy() if torch.is_tensor(theta_all) else np.asarray(theta_all)
    n_features = theta_np.shape[-1]

    totals = {}
    for seed_key, potentials_k in potentials_by_seed.items():
        total = sum(_potential_width(p, k, known_dims)[0] for k, p in potentials_k.items())
        totals[seed_key] = total
        if total == n_features:
            col_map = theta_column_map(potentials_k, n_features, known_dims=known_dims, verbose=False)
            return seed_key, potentials_k, col_map

    raise ValueError(
        f"No seed's fitted potentials sum to theta_all's {n_features} columns.\n"
        f"Per-seed totals: {totals}"
    )


# ---------------------------------------------------------------------
# 2. Load theta for one target potential
# ---------------------------------------------------------------------
def load_theta_for_potential(theta_final, potentials, target_key, known_dims=None, verbose=False):
    """
    Slice theta_t down to the columns of `target_key`.
    Returns (theta_raw: np.ndarray (dim,) or (T, dim), potential, start, dim).
    """
    theta_1_np = theta_final.detach().cpu().numpy() if torch.is_tensor(theta_final) else np.asarray(theta_final)
    col_map = theta_column_map(potentials, len(theta_final), known_dims, verbose)
    # (was `potentials_k` in an earlier edit -- that name isn't in scope here
    # and would NameError; this function only ever sees `potentials`.)
    if target_key not in col_map:
        raise KeyError(f"'{target_key}' not in potentials: {list(potentials.keys())}")
    start, dim = col_map[target_key]

    potential = potentials[target_key]
    if hasattr(potential, 'num_coefficients') and hasattr(potential, 'indices'):
        assert dim == potential.num_coefficients == potential.indices.shape[-1], (
            f"dim mismatch: col_map={dim}, num_coefficients={potential.num_coefficients}, "
            f"indices={potential.indices.shape[-1]}"
        )

    theta_raw = theta_1_np[start:start + dim]
    if verbose:
        print(f"-> {target_key}: cols [{start}:{start + dim}] out of {theta_1_np.shape[-1]}")
    return theta_raw, potential, start, dim


# ---------------------------------------------------------------------
# 3. Decode (j, s1, s2) indices for a 4th-order scattering potential
# ---------------------------------------------------------------------
def _decode_scattering_jsk(potential, verbose=False):
    """
    Shared decoder for Scattering_Fourth_Order_{Real,Imag,Mod2_Real,
    Mod2_Imag}_1d potentials' raw (3, n_coeff) `.indices`. ALL FOUR classes
    build `.indices` via the exact same call,
    indices_fourth_order_Q(J, Q, offset, lite, include_lowpass) (see
    codes/potentials/utils_potentials.py), so they all share the same
    per-column layout regardless of which class produced them:

        col[0], col[1] -- a PAIRED, symmetric pair of FINE, first-layer
            scale indices s1, s2, each in [0, J*Q) -- which two
            (Q-oversampled, sub-octave) first-layer channels' envelope
            fluctuations are being correlated.
        col[2] -- the SHARED, single COARSE second-layer scale j, in
            [0, num_filters) (num_filters = J+1 if that potential's second
            layer includes the low-pass channel, J otherwise) -- the
            resolution at which the s1/s2 correlation is evaluated.

    Verified two independent ways:
      1. Every one of the four classes' fit_micro builds norm_indices as
         self.norm[:,None]*self.norm[None,:] (an outer product over the
         length-J*Q self.norm vector) merely repeated -- NOT varied --
         across a trailing num_filters axis, then indexed with
         indices[0],indices[1],indices[2]. Only col[0]/col[1] can be valid
         indices into that J*Q-sized array (checked below), so those two
         -- not col[2] -- are the fine pair.
      2. Independently confirmed for the Mod2 classes by tracing actual
         forward() tensor shapes (this used to be documented here as
         decode_mod2_stq's own separate finding; it's the same layout).

    A previous version of this decoder (formerly named decode_jab)
    ASSUMED a different layout instead of deriving it: one column was
    guessed to be a shared "first-layer" j, the other two a "second-layer"
    pair s1,s2 > j living in the SAME value range as j. Which column was j
    was picked via a combinatorial coefficient-count fingerprint
    (count(j) == comb(J-j, 2)). That fingerprint is a coincidental
    property of how the (col0,col1,col2) triples happen to enumerate when
    Q=1 (J*Q == J, so the fine and coarse ranges collapse to nearly the
    same size, and grouping any {a<b<c} triple by its minimum trivially
    reproduces that count regardless of which axis is physically shared)
    -- it is not diagnostic of which axis actually broadcasts. Concretely
    this mislabeled every coefficient even at Q=1 (col[0], one fine
    partner, was called "j"; col[1]/col[2] -- one fine, one coarse index --
    were paired as if directly comparable "s1,s2", which then got the
    wrong filter bank in compute_norm_jab) and, for Q>1, no column matched
    the fingerprint at all so it silently fell back to a best guess (40%
    of coefficients violated even that guess's own sanity check in a
    J=4,Q=2 test). Deriving the split directly from J/Q below fixes both.

    Returns j (shape (n_coeff,), in [0, num_filters)), s1, s2 (shape
    (n_coeff,), sorted s1<=s2, both in [0, J*Q)).
    """
    idx = potential.indices.long().detach().cpu().numpy()
    if idx.shape[0] != 3:
        raise ValueError(f"expected potential.indices with 3 rows, got {idx.shape[0]}.")
    J, Q = getattr(potential, 'J', None), getattr(potential, 'Q', None)
    if J is None or Q is None:
        raise RuntimeError("potential is missing J/.Q -- can't validate the fine/coarse index split.")
    num_filters_q = J * Q

    s1_raw, s2_raw, j = idx[0], idx[1], idx[2]
    bad = int((s1_raw >= num_filters_q).sum() + (s2_raw >= num_filters_q).sum())
    if bad:
        raise ValueError(
            f"{bad} entries have col[0]/col[1] >= J*Q={num_filters_q} -- this potential's "
            f".indices doesn't match the (fine pair, coarse singleton) layout that "
            f"indices_fourth_order_Q produces; check potential.J/.Q."
        )
    s1, s2 = np.minimum(s1_raw, s2_raw), np.maximum(s1_raw, s2_raw)
    if verbose:
        num_filters = int(j.max()) + 1 if j.size else 0
        print(f"s1,s2 in [0,{num_filters_q}) (fine, first-layer, J*Q={num_filters_q} channels), "
              f"j in [0,{num_filters}) (coarse, second-layer, shared).")
    return j, s1, s2


def decode_jab(potential, verbose=False):
    """
    Decode (j, s1, s2) for a Scattering_Fourth_Order_{Real,Imag}_1d
    potential (the non-Mod2 classes). Identical layout/logic to
    decode_mod2_stq -- see _decode_scattering_jsk for the full derivation
    and for why a previous version of this function (which guessed the
    layout instead of deriving it) mislabeled the axes for every Q,
    including Q=1. Kept as a separate name for call-site clarity and
    backward compatibility with existing (j, k_prime, k) column naming.

    Returns j, s1, s2 -- see _decode_scattering_jsk.
    """
    return _decode_scattering_jsk(potential, verbose=verbose)


def decode_mod2_stq(potential, verbose=False):
    """
    Decode (j2, s1, s2) for a Scattering_Fourth_Order_Mod2_{Real,Imag}_1d
    potential. Identical layout/logic to decode_jab -- see
    _decode_scattering_jsk for the full derivation (originally established
    here by tracing forward()'s actual tensor shapes: (B, J*Q, J*Q, J+1),
    NOT (B, J+1, J+1, J*Q) as that class's own inline comments claim).

    Returns (j2, s1, s2) -- see _decode_scattering_jsk (j2 is that
    function's `j`).
    """
    return _decode_scattering_jsk(potential, verbose=verbose)


# ---------------------------------------------------------------------
# 3b. Region-pruned scalar potentials (Scalar_GGD_KRegion-style): decode
#     per-coefficient channel index, since these do NOT have a fixed
#     J*Q+1 layout -- num_coefficients = sum over J channels of however
#     many of the K candidate regions survived fitting-time pruning.
# ---------------------------------------------------------------------
def inspect_fitted_scalar_potential(p):
    """
    Print the attributes that plausibly encode per-coefficient channel
    assignment for a region-based scalar potential (K max regions per
    channel, J channels, num_coefficients = sum of active regions across
    channels after fitting-time pruning).

    IMPORTANT: run this on an ACTUALLY FITTED potential (e.g. one seed's
    entry from get_potentials_for_seed(...) / potentials_by_seed), not the
    nominal unfitted one -- Keff/active/active_flat/scale/alpha are only
    populated after fitting, and decode_scalar_channel needs to see real
    values to pick the right decoding path.
    """
    for name in ("K", "J", "num_coefficients", "Keff", "active", "active_flat",
                 "scale", "alpha", "cuts", "pi", "sw", "stat_scale"):
        val = getattr(p, name, "<missing attribute>")
        if val is None or isinstance(val, (int, float, str)):
            print(f"{name:16s} {val}")
        elif torch.is_tensor(val):
            print(f"{name:16s} tensor shape={tuple(val.shape)} dtype={val.dtype}")
        elif isinstance(val, np.ndarray):
            print(f"{name:16s} ndarray shape={val.shape} dtype={val.dtype}")
        else:
            print(f"{name:16s} {type(val).__name__} {val}")


def decode_scalar_channel(potential, verbose=False):
    """
    Per-coefficient channel index for a region-pruned scalar potential.
    Tries, in order:

    1. `active_flat`: an int array of FLAT INDICES into the (J, K) channel
       x region grid -- confirmed against a real fitted potential to have
       length == num_coefficients directly (NOT a boolean mask over the
       full J*K grid, which was this function's first, wrong guess).
       channel = active_flat[i] // K, region = active_flat[i] % K.
    2. `Keff`: per-channel active-region count, length J. Coefficients
       assumed ordered channel-major: Keff[0] coefficients for channel 0,
       then Keff[1] for channel 1, etc. (fallback only -- prefer
       active_flat, which doesn't depend on this ordering assumption at
       all since it gives the channel directly per coefficient).

    NOTE: potential.J here is NOT the number of wavelet octaves -- for a
    real fitted potential it was seen to be J_octaves*Q + 1 (e.g. 9 for a
    Q=1 'morlet' potential, 25 for a Q=3 'psi' potential with J_octaves=8),
    i.e. the same (octave, sub-band, low-pass) channel grid as the
    fixed-layout linear potentials (L_6/L_6_psi), just with per-channel
    region pruning on top.

    Returns channel_indices: int array, shape (num_coefficients,), values
    in [0, potential.J - 1].
    """
    K = getattr(potential, "K", None)
    J = getattr(potential, "J", None)
    n = getattr(potential, "num_coefficients", None)
    if K is None or J is None or n is None:
        raise RuntimeError("potential is missing K / J / num_coefficients -- can't decode channels.")

    active_flat = getattr(potential, "active_flat", None)
    if active_flat is not None:
        af = active_flat.detach().cpu().numpy() if torch.is_tensor(active_flat) else np.asarray(active_flat)
        af = af.reshape(-1)
        if af.shape[0] != n:
            raise ValueError(
                f"active_flat length {af.shape[0]} != num_coefficients={n} -- "
                f"unexpected shape for a flat-index array, inspect the potential directly."
            )
        channel_indices = (af // K).astype(int)
        if channel_indices.size and (channel_indices.max() >= J or channel_indices.min() < 0):
            raise ValueError(
                f"decoded channel indices out of range [0,{J}) -- active_flat may not "
                f"encode flat (channel,region) positions as assumed here."
            )
        if verbose:
            print(f"decode_scalar_channel: used active_flat (flat channel*K+region indices), "
                  f"{n} coefficients across {J} channels.")
        return channel_indices

    Keff = getattr(potential, "Keff", None)
    if Keff is not None:
        keff = Keff.detach().cpu().numpy() if torch.is_tensor(Keff) else np.asarray(Keff)
        if keff.shape[0] != J:
            raise ValueError(f"Keff length {keff.shape[0]} != J={J}.")
        if int(keff.sum()) != n:
            raise ValueError(
                f"sum(Keff)={int(keff.sum())} != num_coefficients={n} "
                f"-- channel-major ordering assumption doesn't hold; inspect the potential directly."
            )
        channel_indices = np.repeat(np.arange(J), keff.astype(int))
        if verbose:
            print(f"decode_scalar_channel: used Keff, {n} coefficients across {J} channels.")
        return channel_indices.astype(int)

    raise RuntimeError(
        "Neither active_flat nor Keff is populated on this potential -- make sure "
        "you're passing a FITTED potential (e.g. one entry from potentials_by_seed), "
        "not the nominal unfitted one. If both ARE populated and this still fails, "
        "run inspect_fitted_scalar_potential(p), share what you see, and adjust the "
        "channel-major ordering assumption above."
    )


# ---------------------------------------------------------------------
# 4. Moment features (for the "physical" normalization theta * m_k)
# ---------------------------------------------------------------------
def compute_mk(potentials, x1):
    mk = []
    with torch.no_grad():
        for p in potentials.values():
            feat = p(x1)
            if feat.ndim == 1:
                feat = feat[:, None]
            mk.append(feat.mean(0))
    return torch.cat(mk).cpu().numpy()


def compute_stdk(potentials, x1, unbiased=True):
    """
    Per-coefficient Std[phi_k(x)] across x1's batch dimension, concatenated
    across potentials in the SAME column order as compute_mk (mirrors it
    exactly, .std() instead of .mean()). Use this together with
    build_all_averaged_vectors -- the same way mk_all/mk_vectors were built
    -- to get std_vectors for
    build_octave_energy_comparison(..., energy_kind='std').

    unbiased: passed to torch.std (ddof=1 sample std if True), matching the
    ddof=1 convention already used elsewhere in this codebase (the
    cross-seed variance/CV/SNR diagnostics). Gives NaN for any potential
    whose feature has a batch dimension of 1 -- same caveat as ddof=1
    anywhere else.
    """
    sk = []
    with torch.no_grad():
        for p in potentials.values():
            feat = p(x1)
            if feat.ndim == 1:
                feat = feat[:, None]
            sk.append(feat.std(0, unbiased=unbiased))
    return torch.cat(sk).cpu().numpy()


# ---------------------------------------------------------------------
# 5a. Tidy dataframe for one REAL-valued potential
# ---------------------------------------------------------------------
def compute_norm_jab(x1, filters, s1_arr, s2_arr, order=2):
    """
    norm_jab[i] = (E[|Wx_s1|^2] * E[|Wx_s2|^2]) ** (order/4), Wx_s = single-
    layer wavelet transform of x1 at absolute scale s, using the same J+1-
    channel `filters` bank the potential's second layer uses.
    order=2 -> sqrt(E1*E2)  (plain Real/Imag potentials, micro-norm on |x|)
    order=4 -> E1*E2        (Mod2 potentials, micro-norm on |x|^2 -- this is
               the one for Scattering_Fourth_Order_Mod2_Real_Q1 /
               ..._Mod2_Imag_Q1, which is what this codebase actually uses)
    Runs on x1's device; filters moved to match, no_grad, no side effects.
    """
    filters = filters.to(x1.device)
    with torch.no_grad():
        Wx_full = torch.fft.ifft(filters * torch.fft.fft(x1))                    # (B, J+1, T)
        E_scale = (Wx_full.abs() ** 2).mean(dim=(0, 2)).detach().cpu().numpy()   # (J+1,)
    s1_arr, s2_arr = np.asarray(s1_arr), np.asarray(s2_arr)
    return (E_scale[s1_arr] * E_scale[s2_arr]) ** (order / 4)


def build_theta_dataframe(theta_final, potentials, target_key, x1, filters,
                           m_k=None, norm_order=None, known_dims=None, verbose=False):
    """
    Builds a tidy dataframe for a single, real-valued target_key. Decodes
    (j, s1, s2) indices if the potential has scattering index structure,
    otherwise builds a generic coefficient table.

    Normalization (theta_t1_normalized):
    - Potentials WITH scattering index structure (.indices): the physical
      micro-norm convention theta_physical = theta_fitted *
      (E[|Wx_s1|^2]*E[|Wx_s2|^2])**(norm_order/4), via compute_norm_jab,
      evaluated at each coefficient's own (s1, s2) scales. decode_jab and
      decode_mod2_stq decode the SAME (fine pair s1,s2 in [0,J*Q); coarse
      shared j in [0,num_filters)) layout for every
      Scattering_Fourth_Order_{Real,Imag,Mod2_Real,Mod2_Imag}_1d potential
      (see _decode_scattering_jsk) -- so compute_norm_jab always evaluates
      against potential.filters_Q (the fine bank), NOT potential.filters,
      since that's the bank fit_micro actually used to build self.norm for
      every one of these classes (verified against its source: self.norm
      has shape (J*Q,), an outer product over that axis, merely broadcast
      -- not varied -- across the coarse axis). Using potential.filters
      here instead would silently compute the wrong micro-norm.
        norm_order=2 -> sqrt(E1*E2)  for plain Real/Imag potentials
        norm_order=4 -> E1*E2        for Mod2_Real/Mod2_Imag potentials
      If norm_order is left as None, it's inferred from the name: 4 if
      'Mod2' is in target_key, else 2 -- pass it explicitly if that
      heuristic is wrong for a given potential's naming.
    - Potentials WITHOUT .indices (L_6, L_2_lowpass, Scalar_*): there's no
      per-coefficient (s1, s2) to feed compute_norm_jab, so this falls back
      to the old theta_fitted * m_k convention if `m_k` is supplied, or NaN
      if it isn't.
    """
    theta_raw, potential, start, dim = load_theta_for_potential(
        theta_final, potentials, target_key, known_dims, verbose
    )
    final_theta = theta_raw[-1] if theta_raw.ndim == 2 else theta_raw

    if hasattr(potential, 'indices'):
        is_mod2 = 'Mod2' in target_key
        j_arr, s1_arr, s2_arr = _decode_scattering_jsk(potential, verbose=verbose)
        pot_filters = getattr(potential, 'filters_Q', filters)

        order = norm_order if norm_order is not None else (4 if is_mod2 else 2)
        norm = compute_norm_jab(x1, pot_filters, s1_arr, s2_arr, order=order)
        final_theta_normalized = final_theta * norm

        df = pd.DataFrame({
            'potential': target_key,
            'coeff_idx': np.arange(dim),
            'j': j_arr, 'k_prime': s1_arr, 'k': s2_arr,
            'theta_t1': final_theta,
            'theta_t1_normalized': final_theta_normalized,
        })
    else:
        if m_k is not None:
            mk_slice = m_k[start:start + dim]
            norm_mk = final_theta * mk_slice
            final_theta_normalized = np.where(norm_mk != 0, norm_mk, np.nan)
        else:
            final_theta_normalized = np.full(dim, np.nan)
        df = pd.DataFrame({
            'potential': target_key,
            'coeff_idx': np.arange(dim),
            'theta_t1': final_theta,
            'theta_t1_normalized': final_theta_normalized,
        })
    return df


# ---------------------------------------------------------------------
# 5b. Tidy dataframe for a REAL+i*IMAG pair, combined as a complex vector
# ---------------------------------------------------------------------
def build_complex_theta_dataframe(theta_final, potentials, real_key, imag_key, x1, filters,
                                   combined_name=None, m_k=None, norm_order=None,
                                   known_dims=None, verbose=False):
    """
    Pairs `real_key` and `imag_key` coefficient-by-coefficient into
    theta_complex = theta_real + i*theta_imag, and returns a tidy dataframe
    with the shared (j, k', k) index structure (taken from the real
    potential).

    Normalization (theta_complex_normalized):
    - If the real potential has scattering index structure (.indices): the
      SAME physical micro-norm convention as build_theta_dataframe --
      theta_physical = theta_complex * (E[|Wx_s1|^2]*E[|Wx_s2|^2])**(norm_order/4),
      via compute_norm_jab, evaluated ONCE at the REAL potential's own
      (s1, s2) scales. If 'Mod2' is in real_key, uses decode_mod2_stq (s1,s2
      are FINE, first-layer indices in [0,J*Q); j (the shared axis) is the
      COARSE, second-layer scale in [0,J]) and pot_re.filters_Q (the fine
      bank -- verified against that class's fit_micro, which builds its own
      micro-norm from filters_Q, not filters). Otherwise uses decode_jab and
      pot_re.filters, falling back to the `filters` argument only if the
      potential doesn't have its own. Because theta_im was already expanded
      onto pot_re's index layout below (using the matching index_convention),
      each complex coefficient's (s1, s2) is shared between its real and
      imaginary parts, so this one real-valued norm array multiplies
      directly into the complex vector -- no separate real/imag norm needed.
        norm_order=2 -> sqrt(E1*E2)  for plain Real/Imag potential pairs
        norm_order=4 -> E1*E2        for Mod2_Real/Mod2_Imag pairs -- this
                         is the one for Scattering_Fourth_Order_Mod2_Real_Q1
                         / ..._Mod2_Imag_Q1, which is what this codebase
                         actually uses.
      If norm_order is left as None, it's inferred from the name: 4 if
      'Mod2' is in real_key, else 2 -- pass it explicitly if that heuristic
      is wrong for a given pair's naming.
    - If the real potential has no .indices: there's no (s1, s2) to feed
      compute_norm_jab, so this falls back to the old
      theta_complex * (mk_re + i*mk_im) convention if `m_k` is supplied
      (mk_re/mk_im sliced the same way theta_re/theta_im were -- valid here
      because reaching this branch already guarantees dim_re == dim_im, see
      below), or leaves theta_complex_normalized as NaN+NaNj if `m_k` isn't
      given either.

    All downstream aggregation (aggregate_theta_across_seeds) takes
    |theta_complex|, i.e. the modulus over BOTH components jointly, not the
    real part alone.
    """
    combined_name = combined_name or real_key.replace('_Real', '').replace('Real_', '')
    is_mod2 = 'Mod2' in real_key

    theta_re, pot_re, start_re, dim_re = load_theta_for_potential(
        theta_final, potentials, real_key, known_dims, verbose)
    theta_im, pot_im, start_im, dim_im = load_theta_for_potential(
        theta_final, potentials, imag_key, known_dims, verbose)

    # The imaginary branch typically has FEWER coefficients than the real one
    # (e.g. diagonal terms are identically zero for Im(z*conj(z)) and get
    # pruned out), so we can't assume matching shapes or identical `.indices`.
    # Expand theta_im onto pot_re's own index layout, zero-filling wherever
    # pot_re has a coefficient pot_im doesn't.
    if hasattr(pot_re, 'indices') and hasattr(pot_im, 'indices'):
        theta_im_expanded = expand_imag_to_real_indices(
            pot_im, pot_re, theta_im, verbose=verbose,
            index_convention='mod2' if is_mod2 else 'jab',
        )
    elif dim_re == dim_im:
        theta_im_expanded = theta_im
    else:
        raise ValueError(
            f"'{real_key}' and '{imag_key}' have no index structure to align on "
            f"(missing .indices) and different sizes ({dim_re} vs {dim_im}) -- "
            f"cannot pair them coefficient-wise."
        )

    theta_re_final = theta_re[-1] if theta_re.ndim == 2 else theta_re
    theta_im_final = theta_im_expanded[-1] if theta_im_expanded.ndim == 2 else theta_im_expanded
    theta_complex = theta_re_final + 1j * theta_im_final

    if hasattr(pot_re, 'indices'):
        j_arr, s1_arr, s2_arr = _decode_scattering_jsk(pot_re, verbose=verbose)
        pot_filters = getattr(pot_re, 'filters_Q', filters)

        order = norm_order if norm_order is not None else (4 if is_mod2 else 2)
        norm = compute_norm_jab(x1, pot_filters, s1_arr, s2_arr, order=order)
        theta_complex_normalized = theta_complex * norm

        df = pd.DataFrame({
            'potential': combined_name,
            'coeff_idx': np.arange(dim_re),
            'j': j_arr, 'k_prime': s1_arr, 'k': s2_arr,
            'theta_complex': theta_complex,
            'theta_complex_normalized': theta_complex_normalized,
        })
    else:
        if m_k is not None:
            # dim_re == dim_im is guaranteed here -- the only way to reach
            # this branch (pot_re has no .indices) without raising above.
            mk_re = m_k[start_re:start_re + dim_re]
            mk_im = m_k[start_im:start_im + dim_im]
            theta_complex_normalized = theta_complex * (mk_re + 1j * mk_im)
        else:
            theta_complex_normalized = np.full(dim_re, np.nan + 1j * np.nan)
        df = pd.DataFrame({
            'potential': combined_name,
            'coeff_idx': np.arange(dim_re),
            'theta_complex': theta_complex,
            'theta_complex_normalized': theta_complex_normalized,
        })
    return df


# ---------------------------------------------------------------------
# 6. Cross-seed aggregation (works for real- or complex-valued columns:
#    np.abs() on a complex array returns the modulus)
# ---------------------------------------------------------------------
def aggregate_theta_across_seeds(dfs, value_col='theta_t1', group_cols=('j', 'k_prime', 'k')):
    """
    dfs: dict {seed_key: df}. Returns tidy df with group_cols +
    ['abs_mean', 'abs_std', 'n_seeds'], where abs_mean/abs_std are computed
    on |value_col| pooled across seeds (and across any index not in
    group_cols). If value_col holds complex numbers, |.| is the modulus.
    """
    group_cols = list(group_cols)
    frames = []
    for seed, df in dfs.items():
        missing = [c for c in group_cols + [value_col] if c not in df.columns]
        if missing:
            raise KeyError(f"seed '{seed}': dataframe missing columns {missing}")
        tmp = df[group_cols + [value_col]].copy()
        tmp['abs_val'] = np.abs(tmp[value_col])
        tmp['seed'] = seed
        frames.append(tmp)
    long = pd.concat(frames, ignore_index=True)

    agg = long.groupby(group_cols)['abs_val'].agg(['mean', 'std', 'count']).reset_index()
    agg = agg.rename(columns={'mean': 'abs_mean', 'std': 'abs_std', 'count': 'n_seeds'})
    agg['abs_std'] = agg['abs_std'].fillna(0.0)
    return agg


def build_agg_cases(cases):
    """cases: dict label -> {'dfs': {seed: df}, 'value_col': str, 'group_cols': tuple (optional)}"""
    out = {}
    for label, spec in cases.items():
        group_cols = spec.get('group_cols', ('j', 'k_prime', 'k'))
        out[label] = aggregate_theta_across_seeds(spec['dfs'], spec['value_col'], group_cols)
    return out


# ---------------------------------------------------------------------
# 7. Plotting -- legends live on each subplot now
# ---------------------------------------------------------------------
def _kprime_colors(k_primes):
    cmap = plt.get_cmap('viridis')
    n = len(k_primes)
    return {kp: cmap(idx / max(1, n - 1) * 0.85) for idx, kp in enumerate(sorted(k_primes))}


def _band_or_line(ax, x, mean, std, n_seeds, color, label, log_y=True):
    ax.plot(x, mean, 'o-', color=color, label=label, ms=4)
    if np.any(np.asarray(n_seeds) > 1):
        if log_y:
            lower = np.clip(mean - std, a_min=mean * 1e-3, a_max=None)
        else:
            lower = mean - std
        upper = mean + std
        ax.fill_between(x, lower, upper, color=color, alpha=0.2, linewidth=0)


def plot_modulus_vs_k_grid(agg_cases, log_y=True, panel_size=(3.2, 3.0)):
    """
    Rows = cases (e.g. MGD / Reg), columns = j. One line per k' per panel,
    legend drawn inside each panel.
    """
    labels = list(agg_cases.keys())
    all_js = sorted(set().union(*[set(df['j'].unique()) for df in agg_cases.values()]))
    all_kprimes = sorted(set().union(*[set(df['k_prime'].unique()) for df in agg_cases.values()]))
    colors = _kprime_colors(all_kprimes)

    n_rows, n_cols = len(labels), len(all_js)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
                              sharex=True, sharey='row', squeeze=False)

    for row, label in enumerate(labels):
        df = agg_cases[label]
        for col, j in enumerate(all_js):
            ax = axes[row, col]
            sub = df[df['j'] == j]
            for kp in all_kprimes:
                g = sub[sub['k_prime'] == kp].sort_values('k')
                if g.empty:
                    continue
                _band_or_line(ax, g['k'].values, g['abs_mean'].values, g['abs_std'].values,
                               g['n_seeds'].values, colors[kp], rf"$k'={kp}$", log_y)
            if log_y:
                ax.set_yscale('log')
            if row == 0:
                ax.set_title(f'j={j}')
            if row == n_rows - 1:
                ax.set_xlabel('k')
            ax.legend(fontsize=7, loc='best')
        axes[row, 0].set_ylabel(f'{label}\n' + r'$|\theta_{j,k^{\prime},k}|$')

    fig.suptitle(r'$|\theta_{j,k^{\prime},k}|$ vs $k$, per $j$')
    plt.tight_layout()
    plt.show()


def plot_modulus_vs_j_grid(agg_cases, log_y=True, panel_size=(3.2, 3.0)):
    """
    Rows = cases, columns = k. One line per k' per panel, x-axis is j.
    Legend drawn inside each panel.
    """
    labels = list(agg_cases.keys())
    all_ks = sorted(set().union(*[set(df['k'].unique()) for df in agg_cases.values()]))
    all_kprimes = sorted(set().union(*[set(df['k_prime'].unique()) for df in agg_cases.values()]))
    colors = _kprime_colors(all_kprimes)

    n_rows, n_cols = len(labels), len(all_ks)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
                              sharex=True, sharey='row', squeeze=False)

    for row, label in enumerate(labels):
        df = agg_cases[label]
        for col, k in enumerate(all_ks):
            ax = axes[row, col]
            sub = df[df['k'] == k]
            for kp in all_kprimes:
                g = sub[sub['k_prime'] == kp].sort_values('j')
                if g.empty:
                    continue
                _band_or_line(ax, g['j'].values, g['abs_mean'].values, g['abs_std'].values,
                               g['n_seeds'].values, colors[kp], rf"$k'={kp}$", log_y)
            if log_y:
                ax.set_yscale('log')
            if row == 0:
                ax.set_title(f'k={k}')
            if row == n_rows - 1:
                ax.set_xlabel('scale j')
            ax.legend(fontsize=7, loc='best')
        axes[row, 0].set_ylabel(f'{label}\n' + r'$|\theta_{j,k^{\prime},k}|$')

    fig.suptitle(r'$|\theta_{j,k^{\prime},k}|$ vs $j$, per $k$')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 8. Readable, averaged-over-seeds theta vectors per potential
# ---------------------------------------------------------------------
def build_all_averaged_vectors(theta_all, potentials, terms, real_imag_pairs=(), known_dims=None):
    """
    theta_all: (n_seeds, n_features) tensor/array of final-time theta,
    stacked across seeds (e.g. `mgd_all` or `reg_all`).
    potentials: the FITTED potentials dict for ONE seed whose column layout
    actually matches theta_all's n_features -- e.g. from
    get_potentials_for_seed(...) / load_potentials_for_config(...), or
    auto-selected via find_matching_reference_potentials(theta_all,
    potentials_by_seed). Do NOT pass the freshly-constructed nominal
    potentials dict straight from get_1d_potentials(...): per-seed fitting
    can prune coefficients (e.g. Scalar_GGD_KRegion-based potentials), so its
    coefficient counts won't generally match theta_all's actual columns.
    terms: full list of potential names, in the same order used to build
    `potentials` (defines the column layout of theta_all).
    real_imag_pairs: list of (real_name, imag_name, combined_name) -- for
    each, the two averaged real vectors are combined into one complex
    vector combined_name = mean(real) + i*mean(imag), and the two raw
    entries are dropped from the returned dict.

    Returns dict: potential_name -> 1D np.ndarray (real or complex),
    averaged over seeds, ready to slice/plot/manipulate directly. E.g.:
        vectors['L_6'], vectors['L_6_psi'], vectors['Scalar_psi_gaussianK'],
        vectors['Scattering_Fourth_Order_Mod2_Q1']  (complex)
    """
    theta_np = theta_all.detach().cpu().numpy() if torch.is_tensor(theta_all) else np.asarray(theta_all)
    n_features = theta_np.shape[-1]
    col_map = theta_column_map(potentials, n_features, known_dims=known_dims, verbose=False)
    mean_theta = theta_np.mean(axis=0)

    vectors = {}
    for name in terms:
        if name not in col_map:
            continue
        start, dim = col_map[name]
        vectors[name] = mean_theta[start:start + dim]

    for real_name, imag_name, combined_name in real_imag_pairs:
        if real_name not in vectors or imag_name not in vectors:
            continue
        pot_re, pot_im = potentials[real_name], potentials[imag_name]
        if hasattr(pot_re, 'indices') and hasattr(pot_im, 'indices'):
            # imag branch is typically shorter (pruned diagonal etc.) --
            # expand it onto the real branch's layout, zero-filling gaps.
            imag_expanded = expand_imag_to_real_indices(
                pot_im, pot_re, vectors[imag_name],
                index_convention='mod2' if 'Mod2' in real_name else 'jab',
            )
        elif vectors[real_name].shape == vectors[imag_name].shape:
            imag_expanded = vectors[imag_name]
        else:
            raise ValueError(f"'{real_name}'/'{imag_name}' shape mismatch and no .indices "
                              f"to align on -- cannot combine.")
        vectors[combined_name] = vectors[real_name] + 1j * imag_expanded
        del vectors[real_name]
        del vectors[imag_name]

    return vectors


# ---------------------------------------------------------------------
# 9. m_k companion vectors + octave-level interacting vs non-interacting
#    energy comparison
#
#    "Non-interacting" = every potential except the 4th-order scattering
#    one: L_6, L_6_psi, L_2_lowpass, Scalar_psi_gaussianK,
#    Scalar_morlet_gaussianK. "Interacting" = the 4th-order scattering
#    (Real + i*Imag, Mod2, Q1) potential.
#
#    The "psi"-family potentials are built with Q=3 (3 sub-bands per
#    octave), the "morlet"-family ones with Q=1, so their raw vectors have
#    different lengths (J*3+1 vs J*1+1). aggregate_octaves_from_QJ collapses
#    away the Q sub-band axis so every potential ends up as a J-length
#    octave profile (+ a separate scalar low-pass), directly comparable.
#    The scattering potential doesn't have this simple per-scale layout --
#    each of its coefficients is a correlation ACROSS two fine, first-layer
#    octaves (s1, s2) evaluated at a shared coarser resolution j (see
#    _decode_scattering_jsk) -- so its "energy" genuinely belongs to BOTH
#    octave(s1) and octave(s2) at once, not to the shared j. This used to
#    be handled by aggregate_by_j grouping on j alone, silently attributing
#    100% of every coefficient's energy to the analysis resolution and 0%
#    to either of the two scales it actually correlates.
#    aggregate_interacting_to_octaves splits each coefficient's energy in
#    half between octave(s1) and octave(s2) instead (both halves land back
#    in the same octave whenever s1, s2 share one -- e.g. two Q-oversampled
#    sub-bands of the same octave -- so that octave still gets full credit).
# ---------------------------------------------------------------------
def aggregate_octaves_from_QJ(vec, J, Q, agg='sum'):
    """
    Collapse a length J*Q+1 coefficient vector -- Q sub-band coefficients
    per octave j=0..J-1 (ordered j-major, index = j*Q + q), followed by one
    trailing low-pass coefficient -- down to a length-J octave profile by
    combining away the Q sub-band axis, plus the low-pass value on its own.

    This is what lets a Q=3 ('psi') potential and a Q=1 ('morlet') potential
    be compared octave-for-octave: after this, both are length J (+ one
    scalar low-pass), regardless of their original Q.

    agg: 'sum' (default) or 'mean'. Use 'sum' when `vec` already holds a
    per-coefficient ENERGY (theta_k * m_k) you want a total for -- sum is
    invariant to how many sub-band coefficients happen to exist, which
    matters once some are pruned away (see aggregate_by_j). Use 'mean' only
    if you specifically want an average coefficient magnitude, not a total.

    ASSUMPTION (flag if it doesn't match your indexing): the vector is
    ordered j-major with Q contiguous sub-band entries per octave, low-pass
    last. If your potentials interleave scales/sub-bands differently, this
    will silently produce a garbage-but-plausible-looking profile -- there's
    no way to detect a wrong ordering convention from the vector alone, so
    it's worth checking against how the filter bank actually indexes things.

    Returns (octaves: shape (J,), lowpass: scalar), dtype matching vec.
    """
    vec = np.asarray(vec)
    expected = J * Q + 1
    if vec.shape[-1] != expected:
        raise ValueError(
            f"expected length J*Q+1 = {J}*{Q}+1 = {expected}, got {vec.shape[-1]} "
            f"-- check J/Q for this potential."
        )
    body, lowpass = vec[:J * Q], vec[J * Q]
    grid = body.reshape(J, Q)
    if agg == 'sum':
        octaves = grid.sum(axis=1)
    elif agg == 'mean':
        octaves = grid.mean(axis=1)
    else:
        raise ValueError(f"agg must be 'sum' or 'mean', got '{agg}'")
    return octaves, lowpass


def aggregate_by_j(vec, j_indices, J, agg='sum'):
    """
    Aggregate a coefficient vector down to J groups by combining all
    entries that share the same integer label `j` (e.g. a scalar
    potential's channel index from decode_scalar_channel; for the
    scattering potential's own (j, s1, s2) triplets, prefer
    aggregate_interacting_to_octaves, which this function backs).

    agg: 'sum' (default) or 'mean'. Use 'sum' when `vec` already holds a
    per-coefficient ENERGY (theta_k * m_k): summing is invariant to how
    many coefficients happen to land in a group, which is exactly what you
    want when e.g. one channel kept 1 active region after fitting-time
    pruning and another kept 4 -- both should count their full
    contribution, not get diluted/inflated by group size. 'mean' is only
    appropriate for raw magnitudes (theta or m_k on their own), and even
    then: averaging theta and averaging m_k separately, then multiplying,
    is NOT the same as summing theta_k*m_k directly (mean(a)*mean(b) !=
    mean(a*b) in general) -- see build_octave_energy_comparison, which
    avoids this by always aggregating the already-multiplied energy.

    Groups with zero matching entries (e.g. a channel whose fit pruned all
    K regions away) come back as 0.0; `counts` lets you tell "genuinely
    zero" apart from "no coefficients in this group".

    Returns (out: shape (J,), counts: shape (J,) int), dtype of `out`
    matches vec (so complex vec -> complex out).
    """
    vec = np.asarray(vec)
    j_indices = np.asarray(j_indices)
    if vec.shape[-1] != j_indices.shape[-1]:
        raise ValueError(f"vec length {vec.shape[-1]} != j_indices length {j_indices.shape[-1]}")

    out = np.zeros(J, dtype=vec.dtype)
    counts = np.zeros(J, dtype=int)
    for v, j in zip(vec, j_indices):
        j = int(j)
        if 0 <= j < J:
            out[j] += v
            counts[j] += 1
        # j outside [0, J) shouldn't happen for a valid index array -- left
        # unguarded-but-silent here since the decoder that produced it
        # (decode_jab / decode_scalar_channel) already warns/raises loudly
        # if its own assumptions look violated.
    if agg == 'sum':
        pass  # out already holds the sum
    elif agg == 'mean':
        nonzero = counts > 0
        out = np.where(nonzero, out / np.maximum(counts, 1), 0.0).astype(vec.dtype)
    else:
        raise ValueError(f"agg must be 'sum' or 'mean', got '{agg}'")
    return out, counts


def aggregate_interacting_to_octaves(vec, s1_indices, s2_indices, Q, J, agg='sum'):
    """
    Attribute a per-coefficient scattering-potential quantity (typically an
    ENERGY theta_k * stat_k) to J octave bins by the two FINE, first-layer
    scales s1, s2 that coefficient actually correlates -- from decode_jab /
    decode_mod2_stq, each in [0, J*Q) -- NOT by the shared coarse scale j.

    A 4th-order scattering coefficient's quantity is a joint statistic of
    TWO scales, octave(s1) and octave(s2) (evaluated at some shared coarser
    resolution j, which is just the analysis window -- see
    _decode_scattering_jsk for why j isn't itself a third energy-bearing
    scale). Crediting it entirely to one arbitrary axis (as grouping by the
    shared j alone does) is wrong for the same reason splitting a
    mode-mode-mode energy-transfer term between only one of the modes would
    be: the effect belongs to both. The standard fix for this kind of cross
    term is to split it evenly between the two scales it connects, so each
    coefficient's energy is added HALF to octave(s1) and HALF to
    octave(s2). When s1 and s2 fall in the SAME octave (possible for Q>1,
    e.g. two sub-bands of one octave), both halves land back in that one
    octave, so it still receives the coefficient's full energy -- as it
    should for a same-octave interaction.

    Unlike aggregate_potential_to_octaves' linear-potential path, there is
    no separate low-pass term here: s1, s2 live in the fine, Q-oversampled
    grid [0, J*Q), which never includes a low-pass channel, so every
    coefficient's energy is fully absorbed into the J octave bins.

    agg: 'sum' (default, for energy) or 'mean' -- passed through to
    aggregate_by_j, which this delegates to (see its docstring for why
    'sum' is the right choice for an already-multiplied energy quantity).

    Returns (octaves: shape (J,), counts: shape (J,) int) -- counts[o] is
    the number of half-contributions landing in octave o (each coefficient
    contributes exactly 2, whether both to one octave or one each to two
    different octaves), for the same "genuinely zero vs no data"
    bookkeeping as aggregate_by_j.
    """
    vec = np.asarray(vec)
    s1_indices = np.asarray(s1_indices)
    s2_indices = np.asarray(s2_indices)
    if not (vec.shape[-1] == s1_indices.shape[-1] == s2_indices.shape[-1]):
        raise ValueError(
            f"vec length {vec.shape[-1]}, s1 length {s1_indices.shape[-1]}, "
            f"s2 length {s2_indices.shape[-1]} must all match."
        )
    octave1 = s1_indices // Q
    octave2 = s2_indices // Q
    half = vec / 2
    return aggregate_by_j(
        np.concatenate([half, half]),
        np.concatenate([octave1, octave2]),
        J, agg=agg,
    )


def aggregate_potential_to_octaves(vec, spec, J, agg='sum'):
    """
    Dispatch to the right octave aggregation depending on `spec`:

      ('QJ', Q)               -- fixed per-scale layout: Q sub-band
                                  coefficients per octave (j-major), one
                                  trailing low-pass coefficient. Use for
                                  dense wavelet-type linear potentials
                                  (L_6, L_6_psi) that don't get pruned.
      ('scalar', potential, Q) -- region-pruned scalar potential
                                  (Scalar_psi_gaussianK,
                                  Scalar_morlet_gaussianK). Two-step
                                  aggregation, both stages using the same
                                  `agg`:
                                    1. collapse surviving regions down to
                                       one value per channel (however many
                                       of the K regions survived
                                       fitting-time pruning for that
                                       channel -- 1, 2, 3, or 4,
                                       independently per channel; channels
                                       with zero surviving regions get
                                       0.0), via decode_scalar_channel +
                                       aggregate_by_j
                                    2. treat that dense per-channel vector
                                       (length potential.J = J*Q+1) exactly
                                       like the 'QJ' case and reuse
                                       aggregate_octaves_from_QJ on it.
                                  Requires potential.J == J*Q+1 -- raises if
                                  the channel grid isn't that shape.

    agg: 'sum' (default) or 'mean' -- passed through to both stages. Use
    'sum' when `vec` is a per-coefficient ENERGY vector (theta_k * m_k):
    that's what makes the result correctly count every active region's
    full contribution regardless of how many regions a given channel's fit
    happened to keep. See aggregate_by_j's docstring for why 'mean' would
    bias channels with different active-region counts against each other.

    Returns (octaves[J], lowpass).
    """
    kind = spec[0]
    if kind == 'QJ':
        _, Q = spec
        return aggregate_octaves_from_QJ(vec, J, Q, agg=agg)
    elif kind == 'scalar':
        _, potential, Q = spec
        J_attr = getattr(potential, 'J', None)
        if J_attr != J * Q + 1:
            raise ValueError(
                f"potential.J={J_attr} != J*Q+1={J}*{Q}+1={J * Q + 1} -- channel layout "
                f"doesn't match a (J octaves x Q sub-bands + 1 low-pass) grid, "
                f"can't apply this decomposition."
            )
        channel_indices = decode_scalar_channel(potential)
        channel_agg, region_counts = aggregate_by_j(np.asarray(vec), channel_indices, J_attr, agg=agg)
        return aggregate_octaves_from_QJ(channel_agg, J, Q, agg=agg)
    else:
        raise ValueError(f"unknown spec kind '{kind}'")


def build_octave_energy_comparison(theta_vectors, stat_vectors, potential_specs,
                                    interacting_key, s1_indices_interacting,
                                    s2_indices_interacting, Q_interacting, J,
                                    scalar_potentials=(), energy_kind='mean'):
    """
    Build a per-octave (length-J) comparison of "interacting" vs
    "non-interacting" energy contributions theta_k * stat_k.

    theta_vectors: dict {potential_name: 1D array}, as returned by
    build_all_averaged_vectors for MGD or Reg.
    stat_vectors: dict {potential_name: 1D array} of the per-coefficient
    feature statistic to multiply theta by. Build it EXACTLY the way
    theta_vectors was built -- build_all_averaged_vectors on the SAME
    reference seed's potentials, SAME real_imag_pairs -- just starting
    from a different raw per-seed quantity:
        energy_kind='mean' (default) -- stat_k = E[phi_k(x)] = m_k,
            from compute_mk(potentials_k, x1)
        energy_kind='std'            -- stat_k = Std[phi_k(x)],
            from compute_stdk(potentials_k, x1)
    theta_vectors and stat_vectors MUST come from the same reference seed:
    the region-pruned Scalar_* potentials have seed-dependent dimension, so
    pairing theta from one seed with stat from another would silently
    misalign coefficients if not caught. Every per-potential product below
    checks theta_vectors[name].shape == stat_vectors[name].shape first and
    raises immediately if they don't match, rather than broadcasting or
    truncating.
    potential_specs: dict {potential_name: spec} for every NON-interacting,
    octave-structured potential to include, where spec is either:
        ('QJ', Q)                  -- e.g. {'L_6': ('QJ', 1), 'L_6_psi': ('QJ', 3)}
        ('scalar', potential, Q)   -- e.g.
            {'Scalar_psi_gaussianK': ('scalar', ref_potentials_mgd['Scalar_psi_gaussianK'], 3),
             'Scalar_morlet_gaussianK': ('scalar', ref_potentials_mgd['Scalar_morlet_gaussianK'], 1)}
    See aggregate_potential_to_octaves for what each spec means.
    interacting_key: name of the real+imag-combined scattering potential in
    theta_vectors/stat_vectors, e.g. 'Scattering_Fourth_Order_Mod2_Q1'.
    s1_indices_interacting, s2_indices_interacting: the fine, first-layer
    (s1, s2) scale arrays for that potential's coefficients, each in
    [0, J*Q) -- decode_mod2_stq(reference_real_potential, verbose=False)[1:]
    (or decode_jab for a non-Mod2 interacting potential). Each coefficient's
    energy is split in half between octave(s1) and octave(s2) -- see
    aggregate_interacting_to_octaves for why (the coefficient is a joint
    statistic of both scales, not of the shared coarse j used only for the
    2nd-layer analysis resolution -- attributing it to j alone silently
    credited a resolution parameter and dropped the two actual scales).
    Q_interacting: that potential's Q (sub-bands per octave), needed to map
    s1/s2 -- indices into the J*Q fine grid -- down to octaves.
    J: number of octaves.
    scalar_potentials: names of potentials with no octave structure at all
    (single-coefficient, e.g. 'L_2_lowpass') -- folded directly into the
    low-pass total rather than an octave profile.
    energy_kind: 'mean' or 'std' -- must match what you actually built
    stat_vectors from; recorded in the output (energy_dict['energy_kind'])
    purely as a label for downstream code/plot titles. It does NOT change
    the arithmetic here (still theta_k * stat_vectors[name][k] either way)
    -- the difference between 'mean' and 'std' energy is entirely in what
    YOU computed stat_vectors from (compute_mk vs compute_stdk).

    Returns dict:
        'non_interacting_by_potential': {name: (octaves[J], lowpass)}
        'non_interacting_total': octaves[J]  -- summed over potentials
        'non_interacting_lowpass_total': scalar -- summed low-pass + scalar terms
        'interacting': octaves[J]  (real-valued) -- each coefficient's
            energy split half-and-half into octave(s1) and octave(s2)
        'interacting_counts': counts[J] -- half-contributions pooled per
            octave (each coefficient contributes 2, to one or two octaves)
        'energy_kind': the energy_kind passed in

    There is no 'interacting_lowpass' term: s1, s2 live in the fine,
    Q-oversampled grid [0, J*Q), which has no low-pass channel, so every
    coefficient's energy is already fully absorbed into the J octave bins.

    "Energy" of a coefficient is theta_k * stat_k for the real potentials,
    and Re(theta_complex * stat_complex) -- plain elementwise complex
    product, then real part, NOT the conjugated Re(theta * conj(stat)) --
    for the interacting one (matches the convention used for
    theta_complex_normalized in build_complex_theta_dataframe; change both
    together if that's not the right convention for your model).

    IMPORTANT: this product is computed at the FINEST available resolution
    (per raw coefficient) FIRST, and only THEN summed within each
    aggregation group (region -> channel, channel/sub-band -> octave, or
    coefficient -> octave for the scattering potential). This matters
    because region counts vary independently per channel (1-4 active
    regions, chosen by fit) and scattering coefficient counts vary per j --
    aggregating theta and stat SEPARATELY and multiplying the aggregates
    afterward would be wrong whenever group sizes vary, since
    mean(theta)*mean(stat) != mean(theta*stat) (and sum(theta)*sum(stat) !=
    sum(theta*stat) either, due to cross terms). Summing raw per-coefficient
    energies is invariant to group size, so it fully and correctly counts
    every active region's contribution regardless of how many regions
    happened to survive pruning at that channel.
    """
    if energy_kind not in ('mean', 'std'):
        raise ValueError(f"energy_kind must be 'mean' or 'std', got '{energy_kind}'")

    def _checked_energy(name):
        if name not in stat_vectors:
            raise KeyError(f"'{name}' not in stat_vectors (have: {list(stat_vectors.keys())})")
        th = np.asarray(theta_vectors[name])
        st = np.asarray(stat_vectors[name])
        if th.shape != st.shape:
            raise ValueError(
                f"'{name}': theta shape {th.shape} != stat_vectors shape {st.shape} -- "
                f"theta_vectors and stat_vectors must be built from the SAME reference "
                f"seed's fitted potentials (Scalar_* potentials have seed-dependent "
                f"dimension after region pruning, so pairing theta from one seed with "
                f"stat from another silently misaligns coefficients unless caught here)."
            )
        return np.real(th * st)

    non_interacting_by_potential = {}
    non_interacting_total = np.zeros(J)
    non_interacting_lowpass_total = 0.0

    for name, spec in potential_specs.items():
        if name not in theta_vectors:
            continue
        energy_raw = _checked_energy(name)
        energy_oct, energy_lp = aggregate_potential_to_octaves(energy_raw, spec, J, agg='sum')
        non_interacting_by_potential[name] = (energy_oct, float(energy_lp))
        non_interacting_total = non_interacting_total + energy_oct
        non_interacting_lowpass_total += float(energy_lp)

    for name in scalar_potentials:
        if name not in theta_vectors:
            continue
        energy = float(_checked_energy(name).sum())
        non_interacting_by_potential[name] = (np.zeros(J), energy)
        non_interacting_lowpass_total += energy

    energy_int_raw = _checked_energy(interacting_key)
    # Split each coefficient's energy in half between the two fine scales
    # (s1, s2) it actually correlates, rather than pooling by the shared
    # coarse j -- see aggregate_interacting_to_octaves.
    interacting, counts = aggregate_interacting_to_octaves(
        energy_int_raw, s1_indices_interacting, s2_indices_interacting,
        Q_interacting, J, agg='sum',
    )

    return {
        'non_interacting_by_potential': non_interacting_by_potential,
        'non_interacting_total': non_interacting_total,
        'non_interacting_lowpass_total': non_interacting_lowpass_total,
        'interacting': interacting,
        'interacting_counts': counts,
        'energy_kind': energy_kind,
    }

def plot_octave_energy_comparison(energy_dict, all_positive = False, title=None):
    """
    Grouped bar plot: summed non-interacting energy vs interacting energy,
    theta_k * stat_k, at each octave j (stat_k = m_k or std_k depending on
    energy_dict['energy_kind']). Low-pass terms aren't shown (they have no
    octave), see energy_dict['non_interacting_lowpass_total'] for that.
    """
    J = len(energy_dict['interacting'])
    x = np.arange(J)
    width = 0.35
    stat_symbol = r'std_k' if energy_dict.get('energy_kind') == 'std' else r'm_k'
    fig, ax = plt.subplots(figsize=(8, 4))
    if all_positive: 
        ax.bar(x - width / 2, np.abs(energy_dict['non_interacting_total']), width,
               label='Non-interacting (linear + scalar)', color='steelblue')
        ax.bar(x + width / 2, np.abs(energy_dict['interacting']), width,
               label='Interacting (4th-order scattering)', color='coral')
        plt.yscale("log") 
    else: 
        ax.bar(x - width / 2, energy_dict['non_interacting_total'], width,
               label='Non-interacting (linear + scalar)', color='steelblue')
        ax.bar(x + width / 2, energy_dict['interacting'], width,
               label='Interacting (4th-order scattering)', color='coral')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('octave j')
    ax.set_ylabel(rf'$\sum_k \theta_k \cdot {stat_symbol}$')
    ax.set_xticks(x)
    ax.legend(loc='best', fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 10. Flat theta-vs-coefficient-index view, per potential -- no
#     (j, k', k) decomposition at all, just theta as its own potential
#     stores it.
# ---------------------------------------------------------------------
def plot_theta_by_potential(vectors, potential_names=None, ncols=3, title=None):
    """
    Grid of small plots, one per potential: theta value vs its own
    coefficient index, in whatever order that potential stores its
    coefficients -- no (j, k', k) decomposition, just the raw
    (already seed-averaged) theta vector as-is.
 
    vectors: dict {potential_name: 1D array}, as returned by
    build_all_averaged_vectors (mgd_vectors / reg_vectors) -- may mix real
    entries (non-scattering: L_6, L_6_psi, L_2_lowpass, Scalar_*, ...) and
    complex ones (the combined scattering potential, e.g.
    Scattering_Fourth_Order_Mod2_Q1). Complex entries are shown as |theta|;
    real entries are shown as signed theta.
    potential_names: which keys to plot, in this order (default: all of
    vectors, in dict order). Pass a subset to show only the non-scattering
    potentials, or only the scattering one -- see usage examples.
    """
    names = potential_names if potential_names is not None else list(vectors.keys())
    missing = [n for n in names if n not in vectors]
    if missing:
        raise KeyError(f"not in vectors: {missing}")
 
    n = len(names)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
 
    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        vec = np.asarray(vectors[name])
        x = np.arange(len(vec))
        if np.iscomplexobj(vec):
            ax.plot(x, np.abs(vec), 'o-', ms=3, color='coral')
            ax.set_ylabel(r'$|\theta|$')
        else:
            ax.plot(x, vec, 'o-', ms=3, color='steelblue')
            ax.axhline(0, color='black', lw=0.5, alpha=0.5)
            ax.set_ylabel(r'$\theta$')
        ax.set_title(f'{name}  (dim={len(vec)})', fontsize=10)
        ax.set_xlabel('coefficient index')
        ax.grid(True, linestyle='--', alpha=0.3)
 
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
 
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 11. Empirical check: does increasing channel index mean coarser or
#     finer scale, for a given filter bank?
# ---------------------------------------------------------------------
def check_filter_scale_ordering(filters, name="filters", verbose=True):
    """
    Empirically checks whether increasing channel index corresponds to a
    COARSER (lower-frequency) or FINER (higher-frequency) scale, for a
    filter bank of shape (1, n_channels, T) IN THE FOURIER DOMAIN (as used
    throughout this codebase -- filters multiply fft(x), they aren't
    convolved in real space).
 
    For each channel, finds the peak-magnitude frequency bin (folding
    negative frequencies onto their positive counterpart, since a
    real-signal filter's magnitude spectrum is symmetric) and reports it as
    a fraction of the Nyquist frequency. If peak frequency decreases
    monotonically as channel index increases: larger index = coarser scale
    (standard wavelet/scattering convention -- also the one decode_jab's
    C(J-j,2) fingerprint implicitly assumes, and what decode_mod2_stq's
    docstring is inferred from). If it increases, this bank uses the
    opposite convention.
 
    Run this on your actual filters/filters_Q tensors (e.g.
    pot_re.filters, pot_re.filters_Q) for a ground-truth answer rather than
    relying on inferred convention.
 
    Returns a DataFrame with columns ['channel', 'peak_freq_frac'].
    """
    f = filters.detach().cpu().numpy() if torch.is_tensor(filters) else np.asarray(filters)
    f = f[0]  # (n_channels, T) -- drop the leading broadcast dim
    n_channels, T = f.shape
    mag = np.abs(f)
    peak_bin = mag.argmax(axis=1)
    peak_bin_folded = np.minimum(peak_bin, T - peak_bin)  # fold to positive-frequency side
    peak_freq_frac = peak_bin_folded / (T // 2)
 
    df = pd.DataFrame({'channel': np.arange(n_channels), 'peak_freq_frac': peak_freq_frac})
    if verbose:
        print(f"{name}: peak frequency (fraction of Nyquist) by channel index")
        print(df.to_string(index=False))
        diffs = np.diff(peak_freq_frac)
        if np.all(diffs <= 1e-9):
            print(f"-> monotonically non-increasing: index 0 = FINEST (highest freq), "
                  f"index {n_channels - 1} = COARSEST (lowest freq / low-pass).")
        elif np.all(diffs >= -1e-9):
            print(f"-> monotonically non-decreasing: index 0 = COARSEST (lowest freq), "
                  f"index {n_channels - 1} = FINEST (highest freq) -- OPPOSITE of the "
                  f"usual convention assumed elsewhere in this module.")
        else:
            print(f"-> NOT monotonic -- inspect peak_freq_frac column manually.")
    return df

# ---------------------------------------------------------------------
# 12. Mean +- std across seeds for one linear (non-pruned) potential,
#     with verified column offsets instead of hardcoded slice indices.
# ---------------------------------------------------------------------
def plot_linear_potential_mean_std(theta_all, potentials, target_key, terms,
                                    label=None, sign=1, known_dims=None,
                                    color='tab:orange', yscale='symlog'):
    """
    Mean +- std across seeds for ONE potential's raw theta_t1 values,
    plotted against its own coefficient index -- for potentials like L_6 /
    L_6_psi that have a fixed (non-pruned) dimension every seed, so slicing
    theta_all by column offset is safe.
 
    theta_all: (n_seeds, n_features) tensor/array (e.g. mgd_all, reg_all).
    potentials: the FITTED potentials dict for a seed whose column layout
    matches theta_all -- use find_matching_reference_potentials to get one,
    same as everywhere else in this module. The offsets for `target_key`
    are read from theta_column_map (NOT hardcoded), so this is safe even if
    a preceding potential's width ever changes.
    sign: 1 or -1 -- multiplies the plotted line AND the shaded std band
    together, so they always represent the same quantity (unlike sign
    flips applied to only one of the two).
    yscale: 'symlog' (default) handles positive AND negative values
    correctly, unlike plain 'log' which silently drops non-positive points
    with no warning -- use 'linear' or 'symlog', not 'log', if theta can be
    negative (check with (theta_all[..., start:start+dim] < 0).any()).
    """
    theta_np = theta_all.detach().cpu().numpy() if torch.is_tensor(theta_all) else np.asarray(theta_all)
    n_features = theta_np.shape[-1]
    col_map = theta_column_map(potentials, n_features, known_dims=known_dims, verbose=False)
    if target_key not in col_map:
        raise KeyError(f"'{target_key}' not in potentials: {list(potentials.keys())}")
    start, dim = col_map[target_key]
 
    theta_slice = sign * theta_np[:, start:start + dim]
    mean = theta_slice.mean(axis=0)
    std = theta_slice.std(axis=0)
 
    x = np.arange(dim)
    label = label or f"{target_key}{' (negated)' if sign == -1 else ''} (mean)"
    plt.figure(figsize=(max(6, dim * 0.3), 4))
    plt.plot(x, mean, marker='o', color=color, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    if yscale != 'linear':
        plt.yscale(yscale)
    plt.title(f'{target_key} (mean +- std across seeds)')
    plt.xlabel('coefficient index')
    plt.ylabel('theta')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
 
    if (theta_slice < 0).any() and yscale == 'log':
        print(f"WARNING: {target_key} has negative values but yscale='log' -- "
              f"those points are invisible on the plot above. Use 'symlog' or 'linear'.")

