"""Hardware-aware runtime configuration for the ML-SDE solvers.

The solver uses small double-precision systems and large CPU-side NumPy
arrays, so CPU BLAS is the safe default even when an accelerator is present.
Runtime choices can be overridden with the ``MLDNN_*`` environment variables
documented beside the corresponding constants below.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def _positive_env(name: str, default: int) -> int:
    """Read a positive integer override, failing early on invalid input."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _available_cpu_count() -> int:
    """Return CPUs available to this process, respecting CPU affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _physical_cpu_count(logical: int) -> int:
    """Best-effort physical/performance-core count without dependencies."""
    if platform.system() == "Darwin":
        try:
            raw = subprocess.check_output(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu_max"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return max(1, min(logical, int(raw.strip())))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    topology = Path("/sys/devices/system/cpu")
    cores: set[tuple[str, str]] = set()
    try:
        allowed = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = set(range(logical))
    for cpu in allowed:
        base = topology / f"cpu{cpu}" / "topology"
        try:
            package = (base / "physical_package_id").read_text().strip()
            core = (base / "core_id").read_text().strip()
            cores.add((package, core))
        except OSError:
            return logical
    return max(1, min(logical, len(cores))) if cores else logical


def _memory_bytes() -> int | None:
    """Return the smaller of host RAM and the active cgroup limit."""
    candidates: list[int] = []
    try:
        candidates.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    limits = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for filename in limits:
        try:
            value = filename.read_text().strip()
            if value != "max":
                limit = int(value)
                if 0 < limit < (1 << 60):
                    candidates.append(limit)
        except (OSError, ValueError):
            continue
    return min(candidates) if candidates else None


def _accelerator() -> tuple[str, str | None, int]:
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return "cuda", props.name, int(props.total_memory)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "Apple Metal", 0
    return "cpu", None, 0


# Hardware ---------------------------------------------------------------------
SYSTEM_OS = platform.system()
MACHINE = platform.machine().lower()
IS_MACOS = SYSTEM_OS == "Darwin"
IS_LINUX = SYSTEM_OS == "Linux"
IS_WINDOWS = SYSTEM_OS == "Windows"
IS_ARM64 = MACHINE in {"arm64", "aarch64"}
IS_WSL = IS_LINUX and (
    "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ
)

CPU_COUNT = _available_cpu_count()
PHYSICAL_CPU_COUNT = _physical_cpu_count(CPU_COUNT)
MEMORY_BYTES = _memory_bytes()
MEMORY_GIB = None if MEMORY_BYTES is None else MEMORY_BYTES / 2**30

# One thread per physical/performance core avoids SMT and nested-BLAS
# oversubscription. Override after benchmarking with MLDNN_NUM_THREADS.
NUM_WORKER_THREADS = min(
    CPU_COUNT, _positive_env("MLDNN_NUM_THREADS", PHYSICAL_CPU_COUNT)
)
TORCH_INTEROP_THREADS = min(
    NUM_WORKER_THREADS,
    _positive_env("MLDNN_INTEROP_THREADS", min(4, NUM_WORKER_THREADS)),
)

# These must be set before importing numerical libraries: BLAS runtimes often
# read their thread limits only once during library initialization.
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_WORKER_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_WORKER_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUM_WORKER_THREADS))

import numpy as np  # noqa: E402  (intentionally imported after thread setup)
import torch  # noqa: E402

ACCELERATOR, ACCELERATOR_NAME, ACCELERATOR_MEMORY_BYTES = _accelerator()

# Small float64 factorizations are faster and more stable on CPU. Explicit
# accelerator overrides remain available for downstream device-safe code.
_device_override = os.getenv("MLDNN_DEVICE", "cpu").lower()
if _device_override == "auto":
    _device_override = ACCELERATOR
if _device_override.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("MLDNN_DEVICE requests CUDA, but CUDA is unavailable")
if _device_override == "mps" and not (
    getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
):
    raise RuntimeError("MLDNN_DEVICE requests MPS, but MPS is unavailable")
DEVICE = _device_override
torch.set_num_threads(NUM_WORKER_THREADS)
try:
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
except RuntimeError:
    # PyTorch permits this only before inter-op work has started.
    pass


# Project paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SOLVER_DIR = PROJECT_ROOT / "solver"
EXPORTS_DIR = PROJECT_ROOT / "exports"
RESULTS_DIR = EXPORTS_DIR / "results"
FIGURES_DIR = EXPORTS_DIR / "figures"
CACHE_DIR = PROJECT_ROOT / "cache"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
SCRATCH_DIR = BENCHMARK_DIR

_library_suffix = ".dylib" if IS_MACOS else ".dll" if IS_WINDOWS else ".so"
LIBFAST_FEM_DYLIB = BENCHMARK_DIR / "libfast_fem.dylib"  # compatibility
LIBFAST_FEM_SO = BENCHMARK_DIR / "libfast_fem.so"  # compatibility
LIBFAST_FEM_PATH = BENCHMARK_DIR / f"libfast_fem{_library_suffix}"

for _directory in (RESULTS_DIR, FIGURES_DIR, CACHE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# Reproducibility and numerical precision --------------------------------------
SEED = 42
TORCH_DTYPE = torch.float64
NUMPY_DTYPE = np.dtype(np.float64)
DPS = 50

# Equation and basis defaults ---------------------------------------------------
ALPHA_DEFAULT = 0.75
ALPHA_SWEEP = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
T_START = 0.0
T_END = 1.0
Y0_DEFAULT = 1.0
MHAT_DEFAULT = 8
MHAT_SWEEP = [4, 6, 8, 10, 12]
NQ_DEFAULT = 64
NQ_SWEEP = [16, 32, 64, 128]
N_MAX_BROWNIAN = 2**16

# Optimization -----------------------------------------------------------------
LAMBDA_B = 1.0
LAMBDA_S = 1.0
GN_MAX_ITER = 50
GN_TOL = 1e-13
LAPACK_DRIVER = "gelsd"

# Reference solvers and Monte Carlo --------------------------------------------
FEM_N_FINE = 2**18
FEM_N_COARSE = 2**14
FEM_N_SWEEP = [2**10, 2**12, 2**14, 2**16, 2**18]
MC_NUM_PATHS = 500
MC_NUM_PATHS_LARGE = 5000

# A float64 fine path is about 2 MiB before work arrays. Keep laptop batches
# conservative and scale automatically on high-memory workstations.
if MEMORY_GIB is None:
    _adaptive_batch = 64
elif MEMORY_GIB < 6:
    _adaptive_batch = 32
elif MEMORY_GIB < 16:
    _adaptive_batch = 64
elif MEMORY_GIB < 64:
    _adaptive_batch = 128
else:
    _adaptive_batch = 256
MC_BATCH_SIZE = _positive_env(
    "MLDNN_MC_BATCH_SIZE", min(MC_NUM_PATHS, _adaptive_batch)
)


@dataclass(frozen=True)
class HardwareProfile:
    os: str = SYSTEM_OS
    machine: str = MACHINE
    is_wsl: bool = IS_WSL
    logical_cpus: int = CPU_COUNT
    physical_cpus: int = PHYSICAL_CPU_COUNT
    memory_bytes: int | None = MEMORY_BYTES
    accelerator: str = ACCELERATOR
    accelerator_name: str | None = ACCELERATOR_NAME
    accelerator_memory_bytes: int = ACCELERATOR_MEMORY_BYTES
    compute_threads: int = NUM_WORKER_THREADS
    interop_threads: int = TORCH_INTEROP_THREADS
    monte_carlo_batch_size: int = MC_BATCH_SIZE


HARDWARE = HardwareProfile()


@dataclass
class BasisConfig:
    alpha: float = ALPHA_DEFAULT
    mhat: int = MHAT_DEFAULT
    nq: int = NQ_DEFAULT
    n_max_brownian: int = N_MAX_BROWNIAN
    dps: int = DPS


@dataclass
class SolverOptConfig:
    lambda_b: float = LAMBDA_B
    lambda_s: float = LAMBDA_S
    gn_max_iter: int = GN_MAX_ITER
    gn_tol: float = GN_TOL
    lapack_driver: str = LAPACK_DRIVER


@dataclass
class FEMConfig:
    n_fine: int = FEM_N_FINE
    n_coarse: int = FEM_N_COARSE
    num_threads: int = NUM_WORKER_THREADS
    lib_path: Path = LIBFAST_FEM_PATH
    use_c_extension: bool = field(default_factory=LIBFAST_FEM_PATH.exists)


@dataclass
class SolverConfig:
    seed: int = SEED
    device: str = DEVICE
    dtype: torch.dtype = TORCH_DTYPE
    t_span: tuple[float, float] = (T_START, T_END)
    y0: float = Y0_DEFAULT
    hardware: HardwareProfile = field(default_factory=lambda: HARDWARE)
    basis: BasisConfig = field(default_factory=BasisConfig)
    opt: SolverOptConfig = field(default_factory=SolverOptConfig)
    fem: FEMConfig = field(default_factory=FEMConfig)

    def __post_init__(self) -> None:
        torch.set_default_dtype(self.dtype)


DEFAULT_CONFIG = SolverConfig()


def hardware_summary() -> str:
    """Return a compact diagnostic string for logs and bug reports."""
    memory = "unknown RAM" if MEMORY_GIB is None else f"{MEMORY_GIB:.1f} GiB RAM"
    accelerator = ACCELERATOR_NAME or ACCELERATOR.upper()
    return (
        f"{SYSTEM_OS}/{MACHINE}{' (WSL)' if IS_WSL else ''}; "
        f"{PHYSICAL_CPU_COUNT} physical/{CPU_COUNT} logical CPUs; {memory}; "
        f"{accelerator}; {NUM_WORKER_THREADS} compute threads; "
        f"MC batch {MC_BATCH_SIZE}"
    )
