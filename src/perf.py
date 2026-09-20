"""CPU thread configuration for training runs.

Import this module **first** — before numpy / torch / stable-baselines3 — so the
BLAS limits below are set before those libraries freeze their thread-pool sizes
at import time.

The problem it solves: with SubprocVecEnv, each of the N env workers otherwise
spins up a full BLAS thread pool for its numpy preprocessing. On a 16-core box
with 8 envs that is 8 pools fighting over 16 cores during rollout collection —
oversubscription that makes collection *slower*, not faster. Capping the workers
at one BLAS thread each fixes that.

Since PPO is synchronous (envs step, then the learner trains — never at the same
time), the learner can then claim all the cores during its training phase, which
setup_cpu_threads restores via torch.set_num_threads.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def setup_cpu_threads(torch_threads=None):
    """Give this process (the PPO learner) its torch threads back. Call once in
    main(), after argument parsing.

    None defaults to *physical* cores (cpu_count // 2): on this box torch at the
    full 16 logical CPUs oversubscribes the 8 physical cores and is ~12% slower
    than 8. Override with --torch-threads to re-tune on other hardware."""
    import torch
    n = torch_threads if torch_threads and torch_threads > 0 else max(os.cpu_count() // 2, 1)
    torch.set_num_threads(n)
    print(f"[perf] learner torch threads={torch.get_num_threads()}  "
          f"env-worker BLAS threads={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    return n
