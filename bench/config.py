"""Fixed experiment constants.

Nothing here is tuned on the test cases: values are chosen a priori and shared by
every problem. Change them only between experiments, never within one.
"""

from __future__ import annotations

SEED = 42

# Benchmark grid
FAMILIES = ("exp_mixture", "sin_mixture", "coupled_tanh")
PARAM_COUNTS = (2, 8, 32, 96)

# Time budget (milliseconds)
BUDGETS_MS = (1, 5, 10, 25)
RUN_DEADLINE_MS = 25.0

# Success targets on normalised MSE
TARGETS = (1e-6, 1e-10)

# Global limits (seconds). The worker stops launching jobs cooperatively at
# WORKER_DEADLINE_S; the parent kills it unconditionally at WATCHDOG_S.
WORKER_DEADLINE_S = 7.5
WATCHDOG_S = 9.0

# Near starts: theta_true + NEAR_SCALE * box_width * N(0, 1)
NEAR_SCALE = 0.05

# Mixtures with p >= this are reported as the "high-order mixture" (poorly
# identifiable) regime, separately from the well-posed cases.
HIGH_ORDER_MIXTURE_P = 32

# Threading environment applied to the worker before numpy is imported
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
