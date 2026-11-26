"""Generate data/examples/damped_curves.csv: 12 noisy damped-oscillation curves (long layout).

y = A exp(-k x) cos(w x) + c + noise,  noise ~ N(0, sigma), sigma = 0.02
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
x = np.linspace(0, 4, 60)
rows = []
for i in range(12):
    A, k, w, c = (
        rng.uniform(0.8, 2.0),
        rng.uniform(0.2, 1.0),
        rng.uniform(1.5, 5.0),
        rng.uniform(-0.3, 0.3),
    )
    y = A * np.exp(-k * x) * np.cos(w * x) + c + rng.normal(0, 0.02, x.size)
    rows.append(
        pd.DataFrame({"curve": f"c{i:02d}", "x": x.round(6), "y": y.round(6), "sigma": 0.02})
    )
out = Path(__file__).resolve().parents[1] / "data/examples/damped_curves.csv"
pd.concat(rows).to_csv(out, index=False)
print(f"wrote {out}")
