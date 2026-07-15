from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import pywt

from torchvision.transforms import v2
from scipy.special import erfcx, erf, erfinv
from scipy.ndimage import gaussian_filter
from scipy import io


# -------- 1d signals --------

def load_SNP(nb_copy=1):
    return torch.load('../../data/data_files/SNP')[None, None].repeat(nb_copy,1,1)

def load_turbulence_1d(sous_ech = 0):


    data_path = '/lustre/fswork/projects/rech/wbg/ukv59en/conditional_mgd/data/data_files/turbulence_1d_period.pt'
    
    Data = torch.load(data_path)
    Data = Data.to(torch.float32)
  
    for i in range(sous_ech):
        if Data.shape[-1]%2 == 0:
            Data = (Data[:,:,::2]+Data[:,:,1::2]) / 2
        else:
            Data = (Data[:,:,:-1:2]+Data[:,:,1::2]) / 2
  
    return (Data)

# -------- 2d signals --------

def load_quijote(fact=0):

    if fact%2 != 0:
        print("Parameter 'fact' should be a power of 2.")
        return
    
    data_quijote = np.load('../../data/data_files/Quijote_Fidu_15000_256.npy')[:1000]
    
    data_quijote -= data_quijote.min()
    data_quijote = np.log(data_quijote+1e-1)

    if fact != 0:
        data_quijote_downsampled = torch.zeros((data_quijote.shape[0],1,data_quijote.shape[-2]//fact,data_quijote.shape[-1]//fact))

        for i in range(data_quijote.shape[0]):
            data_quijote_downsampled[i] = torch.from_numpy(pywt.wavedec2(data_quijote[i], 'db4', mode='periodization', level=int(np.log2(fact)))[0])

        data_quijote = data_quijote_downsampled
    else:
        data_quijote = torch.from_numpy(data_quijote)[:,None]
    
    return data_quijote#torch.log(data_quijote+1e-2)"""


def load_turbulence_2D(fact=0):

    if fact%2 != 0:
        print("Parameter 'fact' should be a power of 2.")
        return

    data_turbulence = io.loadmat("../../data/data_files/ns_randn4_N256_c1.mat")['imgs'].T[:,None]

    if fact != 0:
        DATA_ = data_turbulence

        DATA = torch.zeros(DATA_.shape[0],1,DATA_.shape[-2]//fact,DATA_.shape[-1]//fact)
        for i in range(DATA_.shape[0]):
            DATA[i,0] = torch.from_numpy(pywt.wavedec2(DATA_[i,0], 'db4', mode='periodization', level=int(np.log2(fact)))[0])
    else:
        DATA = torch.from_numpy(data_turbulence).float()
        
    return DATA


"""
Loader for the Chanal et al. (2000) low-temperature gaseous helium jet
1D turbulence time series.

Notebook usage
--------------
Everything below (Segment, HeliumJetRun, HeliumJetDataset, ...) is internal
plumbing. From a notebook, you only need:

    from data_loader import load_reynolds

    velocity = load_reynolds(208)   # ndarray, shape (n_files, T), float32, m/s

    import matplotlib.pyplot as plt
    plt.plot(velocity[0, :5000])

If you're feeding this into a torch pipeline (e.g. `split_periodize_reshape`,
wavelet scattering, `W.decompose`), use the torch entry point instead --
it already has the channel dim so you don't need `velocity[:, None, :]`:

    from data_loader import load_reynolds_torch

    velocity = load_reynolds_torch(208, device="cuda")  # (n_files, 1, T)
    x = split_periodize_reshape(velocity, 1024)

If you also want the physical metadata (dt, viscosity, Rlambda), that's a
separate call so the array stays a plain array:

    from data_loader import run_info
    run_info(208)   # {"rlambda": 208, "dt_s": 20.691e-06, ...}

Design rationale
-----------------
The raw data is a set of independent acquisition files (one per .gz in the
README, e.g. R208_1, R208_2, R208_3 for the Rlambda=208 run). Each file is a
*single continuous* hot-wire recording. Concatenating across files (as the
line-counter in the original loader implicitly did whenever a trajectory of
length T didn't fit evenly in the remaining lines of a file) mixes samples
from two physically distinct acquisitions into one "trajectory", which is
wrong: there is no reason to expect continuity (in time, in calibration, or
even in strict stationarity) across a file boundary.

This module instead:
  1. Loads each file into its own `Segment` (metadata + 1D array), never
     splicing two files together.
  2. Groups segments sharing a Reynolds number into a `HeliumJetRun`, which
     also carries the physical constants for that acquisition campaign
     (Taylor-microscale Reynolds number, kinematic viscosity, sampling step)
     taken from the README files (HYDRO_1, HYDRO_2, HYDRO_5, ...).
  3. Segments are lazy and cache themselves to disk as mmap'd `.npy` on
     first read, so repeated access doesn't reparse multi-GB text files.
  4. `load_reynolds(reynolds)` is the one function meant to be imported
     elsewhere -- it returns plain, already-materialized numpy arrays.

Fill in the metadata dicts below (`viscosity`, exact `rlambda`, README
name/date) from your local README files -- the placeholders here reproduce
what's visible in your original script plus the HYDRO_1/2/5 excerpts you
quoted.
"""

DATA_PATH = Path(__file__).parent / "data_files"
R208_703 = DATA_PATH / "helium_jets" / "R208-703"
R89_929 = DATA_PATH / "helium_jets" / "R89-929"

# ---------------------------------------------------------------------------
# Run metadata (one entry per Reynolds number / acquisition campaign)
# ---------------------------------------------------------------------------
# `files`   : acquisition files for this run, in original order
# `lengths` : number of samples in each file (used only as a sanity check
#             against what's actually read, not for indexing logic)
# `rlambda` : Taylor-microscale Reynolds number as given in the README
# `viscosity_m2_s` : kinematic viscosity in m^2/s, from the README
# `dt_s`    : sampling step in seconds
# `readme`  : free-text provenance (HYDRO folder name + date), for traceability
HELIUM_JET_RUNS: dict[int, dict] = {
    89: {
        "files": [R89_929 / "R89_1", R89_929 / "R89_2",
                  R89_929 / "R89_3"], # there would be a last one but it's not usable 
        "lengths": [9914689, 8758242, 8234387, 9203942],
        "usable_lengths": [
            8_500_000,
            8_500_000,
            8_500_000,
        ],
        "rlambda": 89,
        "viscosity_m2_s": None,   # TODO: fill from the relevant HYDRO_* README
        "dt_s": 47.47e-6,
        "readme": "TODO: HYDRO_? folder / date for R89",
    },
    208: {
        "files": [R208_703 / "R208_1", R208_703 / "R208_2", R208_703 / "R208_3"],
        "lengths": [16798538, 16714211, 16743013],
        "usable_lengths": [
            7_000_000,
            7_000_000,
            7_000_000,
        ],
        "rlambda": 208,
        "viscosity_m2_s": 7.17e-8,
        "dt_s": 20.691e-6,
        "readme": "HYDRO_1, 24/07/97",
    },
    463: {
        "files": [R208_703 / "R463_1", R208_703 / "R463_2",
                  R208_703 / "R463_3", R208_703 / "R463_4"],
        # README lists R463_1..R463_7; add the remaining files/lengths here
        # if you have them (R463_5, R463_6, R463_7).
        "lengths": [16737738, 16746868, 16779570, 16710020],
        "usable_lengths": [
            7_000_000,
            7_000_000,
            7_000_000,
            7_000_000,
        ],
        "rlambda": 463,
        "viscosity_m2_s": 7.20e-8,
        "dt_s": 6.514e-6,
        "readme": "HYDRO_2, 25/07/97",
    },
    703: {
        "files": [R208_703 / "R703_1", R208_703 / "R703_2",
                  R208_703 / "R703_3", R208_703 / "R703_4"],
        "lengths": [9167884, 9457329, 9479764, 9500737],
        "usable_lengths": [
            9_000_000,
            9_000_000,
            9_000_000,
            9_000_000,
        ],
        "rlambda": 703,
        "viscosity_m2_s": 7.850e-8,
        "dt_s": 5.69e-6,
        "readme": "HYDRO_5, 09/08/97",
    },
    929: {
        "files": [R89_929 / "R929_1", R89_929 / "R929_2",
                  R89_929 / "R929_3", R89_929 / "R929_4"],
        "lengths": [8847575, 8750946, 8743745, 8753364],
        "usable_lengths": [
            7_500_000,
            7_500_000,
            7_500_000,
            7_500_000,
        ],
        "rlambda": 929,
        "viscosity_m2_s": None,  # TODO: fill from README
        "dt_s": 6.88e-6,         # NB: original code had `6.88 - 6` (a bug)
        "readme": "TODO: HYDRO_? folder / date for R929",
    },
}

# Conversion factor used in the original code: raw samples -> m/s
# ("according to Muzy, data requires to be divided by 1e5")
VELOCITY_SCALE = 1e5


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """One continuous acquisition file: a single physical recording.

    Lazy by design: nothing is read from disk until `.x` (or `.windows`,
    or `.n_samples` if `expected_length` is unknown) is accessed. Once
    read, the data is cached on disk as a memory-mapped `.npy` next to a
    `cache_dir`, so subsequent access -- even in a new process -- is a
    cheap mmap rather than a re-parse of a multi-GB text file.
    """
    path: Path
    rlambda: int
    dt_s: float
    file_index: int          # position within the run's file list
    cache_dir: Optional[Path] = None
    expected_length: Optional[int] = None  # from README table, no I/O needed
    usable_length: Optional[int] = None
    _x: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.path.stem}.npy"

    def _load(self) -> np.ndarray:
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            return np.load(cache_path, mmap_mode="r")

        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as f:
            arr = np.loadtxt(f, dtype=np.float64)
        arr = (arr / VELOCITY_SCALE).astype(np.float32)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, arr)
            arr = np.load(cache_path, mmap_mode="r")  # reopen as mmap
        return arr

    @property
    def x_raw(self) -> np.ndarray:
        """Full acquisition, no truncation."""
        if self._x is None:
            self._x = self._load()
        return self._x

    @property
    def x(self) -> np.ndarray:
        """Loads (or mmaps, if cached) on first access."""
        if self._x is None:
            self._x = self._load()

            if self.usable_length is not None:
                self._x = self._x[:self.usable_length]

        return self._x

    @property
    def is_loaded(self) -> bool:
        return self._x is not None

    @property
    def n_samples(self) -> int:
        if not self.is_loaded:
            if self.usable_length is not None:
                return self.usable_length
            if self.expected_length is not None:
                return self.expected_length

        return self.x.shape[0]

    def windows(self, T: int, stride: Optional[int] = None) -> np.ndarray:
        """Non-overlapping (or strided) length-T windows *within this file
        only*. Triggers a load/mmap of this segment. Returns shape
        (n_windows, T)."""
        stride = stride or T
        n = (self.n_samples - T) // stride + 1
        if n <= 0:
            return np.empty((0, T), dtype=np.float32)
        idx = np.arange(n) * stride
        # np.stack materializes the windows in RAM (that's the point --
        # this is the "chosen" subset), but self.x itself stays mmap'd.
        return np.stack([np.asarray(self.x[i:i + T]) for i in idx], axis=0)

    def __repr__(self):
        loaded = "loaded" if self.is_loaded else "not loaded"

        return (
            f"Segment("
            f"path={self.path.name!r}, "
            f"rlambda={self.rlambda}, "
            f"expected={self.expected_length}, "
            f"usable={self.usable_length}, "
            f"{loaded})"
        )


@dataclass
class HeliumJetRun:
    """All segments sharing one Reynolds number, plus shared physical constants."""
    rlambda: int
    dt_s: float
    viscosity_m2_s: Optional[float]
    readme: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def total_samples(self) -> int:
        return sum(s.n_samples for s in self.segments)

    def summary(self) -> str:
        lines = [
            f"Rlambda={self.rlambda}  dt={self.dt_s}s  "
            f"nu={self.viscosity_m2_s} m^2/s  ({self.readme})"
        ]
        for s in self.segments:
            lines.append(f"  [{s.file_index}] {s!r}")
        lines.append(f"  total_samples (README) = {self.total_samples}")
        return "\n".join(lines)

    def get_windows(
        self,
        T: int,
        B: Optional[int] = None,
        stride: Optional[int] = None,
        segment_indices: Optional[list[int]] = None,
        mode: Literal["sequential", "random"] = "sequential",
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Extract length-T windows, never spanning a file boundary.

        Returns
        -------
        x : ndarray, shape (n, 1, T)
        provenance : list of (segment_index, start_sample) for each window,
            so you always know which acquisition file (and where in it)
            each window came from.
        """
        segs = self.segments if segment_indices is None else [
            self.segments[i] for i in segment_indices
        ]

        all_windows = []
        provenance: list[tuple[int, int]] = []
        for seg in segs:
            w = seg.windows(T, stride=stride)
            starts = (np.arange(w.shape[0]) * (stride or T))
            all_windows.append(w)
            provenance.extend((seg.file_index, int(s)) for s in starts)

        x = np.concatenate(all_windows, axis=0) if all_windows else np.empty((0, T))

        if mode == "random":
            rng = rng or np.random.default_rng()
            order = rng.permutation(x.shape[0])
            x = x[order]
            provenance = [provenance[i] for i in order]

        if B is not None:
            x = x[:B]
            provenance = provenance[:B]

        return x[:, None, :], provenance


DEFAULT_CACHE_DIR = DATA_PATH / "turbulence_1d" / "_npy_cache"


@dataclass
class HeliumJetDataset:
    """Top-level access point covering every Rlambda run.

    `load_all()` (or `load(reynolds=...)` for a subset) only builds the
    metadata structure -- Segments know their path, expected length, and
    physical constants, but no file is opened and nothing is read into
    memory. Actual data only touches disk the moment you access
    `segment.x`, call `.windows(...)`, or `.get_windows(...)` on a run --
    at which point it's read once and cached as an mmap'd `.npy`, so a
    second look (even a different subset of samples) doesn't reparse the
    original multi-GB text file.

    This lets you build the whole dataset up front and *then* decide,
    interactively, what to actually load:
        ds = HeliumJetDataset.load_all()
        print(ds.summary())          # no I/O
        ds[703].segments[0].x        # first real read of that one file
    """
    runs: dict[int, HeliumJetRun] = field(default_factory=dict)
    cache_dir: Optional[Path] = None

    @classmethod
    def load_all(cls, cache_dir: Optional[Path] = DEFAULT_CACHE_DIR) -> "HeliumJetDataset":
        """Build the full dataset (all Rlambda runs), lazily. No file I/O."""
        return cls.load(reynolds=list(HELIUM_JET_RUNS), cache_dir=cache_dir)

    @classmethod
    def load(
        cls,
        reynolds: int | list[int],
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    ) -> "HeliumJetDataset":
        """Build the dataset structure for one or several runs, lazily.
        No file I/O happens here -- see class docstring."""
        reynolds_list = [reynolds] if isinstance(reynolds, int) else reynolds
        ds = cls(cache_dir=cache_dir)
        for r in reynolds_list:
            ds.runs[r] = ds._build_run(r)
        return ds

    def _build_run(self, reynolds: int) -> HeliumJetRun:
        if reynolds not in HELIUM_JET_RUNS:
            raise ValueError(
                f"Unknown reynolds number {reynolds}. "
                f"Available: {sorted(HELIUM_JET_RUNS)}"
            )
        meta = HELIUM_JET_RUNS[reynolds]
        run = HeliumJetRun(
            rlambda=meta["rlambda"],
            dt_s=meta["dt_s"],
            viscosity_m2_s=meta["viscosity_m2_s"],
            readme=meta["readme"],
        )

        usable_lengths = meta.get("usable_lengths", meta["lengths"])

        for i, (path, expected_len, usable_len) in enumerate(
            zip(
                meta["files"],
                meta["lengths"],
                usable_lengths,
            )
        ):
            run.segments.append(
                Segment(
                    path=path,
                    rlambda=reynolds,
                    dt_s=meta["dt_s"],
                    file_index=i,
                    cache_dir=self.cache_dir,
                    expected_length=expected_len,
                    usable_length=usable_len,
                )
            )

        return run

    def summary(self) -> str:
        """Pure metadata report -- safe to call before loading anything."""
        parts = [self.runs[r].summary() for r in sorted(self.runs)]
        return "\n\n".join(parts)

    def preload(self, reynolds: Optional[int | list[int]] = None) -> None:
        """Explicitly force-read (or mmap-cache) segments. Opt-in only --
        call this once you've decided what you actually need."""
        targets = (
            list(self.runs) if reynolds is None
            else [reynolds] if isinstance(reynolds, int)
            else reynolds
        )
        for r in targets:
            for seg in self.runs[r].segments:
                _ = seg.x  # triggers load/mmap + cache write

    def __getitem__(self, reynolds: int) -> HeliumJetRun:
        return self.runs[reynolds]

    def __repr__(self) -> str:
        return (f"HeliumJetDataset(runs={sorted(self.runs)}, "
                f"cache_dir={self.cache_dir})")


def load_reynolds(
    reynolds: int,
    sous_ech: int = 0,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """Load the velocity time series for one Reynolds number.

    Returns a single ndarray of shape (n_files, T), float32, velocity in
    m/s -- one row per acquisition file, truncated to the shortest file's
    length so they stack into a rectangular array. This is the thing you
    plot: `plt.plot(velocity[0])`.

    `sous_ech` follows the same convention as your `load_turbulence_1d`:
    each unit halves the length by averaging adjacent pairs of samples
    (dt effectively doubles each time).

    Example
    -------
    >>> velocity = load_reynolds(208)
    >>> velocity.shape
    (3, 16714211)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(velocity[0, :5000])
    """
    run = HeliumJetDataset.load(reynolds=reynolds, cache_dir=cache_dir)[reynolds]

    arrays = [np.asarray(seg.x) for seg in run.segments]  # materialize, plain ndarray
    T = min(a.shape[0] for a in arrays)
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) > 1:
        print(f"[load_reynolds] files have different lengths {sorted(lengths)}, "
              f"truncating all to {T}")
    velocity = np.stack([a[:T] for a in arrays], axis=0)

    for _ in range(sous_ech):
        if velocity.shape[-1] % 2 == 0:
            velocity = (velocity[:, ::2] + velocity[:, 1::2]) / 2
        else:
            velocity = (velocity[:, :-1:2] + velocity[:, 1::2]) / 2

    return velocity.astype(np.float32)


def load_reynolds_concatenated(
    reynolds: int,
    sous_ech: int = 0,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Same data as `load_reynolds`, but flattened into one 1D array,
    plus a matching integer array telling you which file each sample
    came from (so concatenation is visible, not hidden).

    Returns (x, file_id): both 1D, same length.
    """
    velocity = load_reynolds(reynolds, sous_ech=sous_ech, cache_dir=cache_dir)
    n_files, T = velocity.shape
    x = velocity.reshape(-1)
    file_id = np.repeat(np.arange(n_files, dtype=np.int32), T)
    return x, file_id


def load_reynolds_torch(
    reynolds: int,
    sous_ech: int = 0,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    device: str = "cpu",
) -> "torch.Tensor":
    """Same data as `load_reynolds`, but as a torch.float32 tensor with a
    channel dim already in place: shape (n_files, 1, T) -- matching what
    your `load_turbulence_1d` pipeline (and `split_periodize_reshape`)
    expects, so you can skip the manual `velocity[:, None, :]` step.

    Example
    -------
    >>> velocity = load_reynolds_torch(208)
    >>> velocity.shape
    torch.Size([3, 1, 16714211])
    >>> x = split_periodize_reshape(velocity, 1024)
    """
    if torch is None:
        raise ImportError("torch is not installed in this environment")
    velocity = load_reynolds(reynolds, sous_ech=sous_ech, cache_dir=cache_dir)
    return torch.from_numpy(velocity).unsqueeze(1).to(device=device, dtype=torch.float32)



def run_info(reynolds: int) -> dict:
    """Physical metadata for a run (dt, Rlambda, viscosity, README
    provenance) -- separate from the array data since you rarely need
    both at once."""
    run = HeliumJetDataset.load(reynolds=reynolds)[reynolds]
    return {
        "rlambda": run.rlambda,
        "dt_s": run.dt_s,
        "viscosity_m2_s": run.viscosity_m2_s,
        "readme": run.readme,
        "file_names": [s.path.name for s in run.segments],
    }


if __name__ == "__main__":
    # Quick sanity check when running `python data_loader.py` directly.
    velocity = load_reynolds(208)
    print("velocity shape:", velocity.shape, velocity.dtype)
    print("mean/std:", velocity.mean(), velocity.std())
    print(run_info(208))