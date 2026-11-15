"""Launches drivestudio's tools/train.py without modifying drivestudio.

The initial background Gaussian scale is the log of the mean distance to the 3 nearest neighbours. Duplicate points give
distance 0 -> log(0) = -inf, so sharp_shape_reg becomes 0/0 = NaN and training stops at step 0 (reproduced on scene-1077).
RigidNodes already clamps the same value to (0.002, 100); the same lower bound is applied to the background.

Usage: python train_launcher.py --min_init_dist 0.002 -- <tools/train.py arguments...>
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

DS = Path(__file__).resolve().parents[2] / "third_party" / "drivestudio"


def main() -> None:
    argv = sys.argv[1:]
    sep = argv.index("--")
    min_dist = float(argv[argv.index("--min_init_dist") + 1]) if "--min_init_dist" in argv[:sep] else None
    train_args = argv[sep + 1:]
    sys.path.insert(0, str(DS))
    if min_dist is not None:
        import models.gaussians.vanilla as vanilla

        orig = vanilla.k_nearest_sklearn

        def clamped(x, k):
            d, i = orig(x, k)
            return d.clip(min=min_dist), i

        vanilla.k_nearest_sklearn = clamped
    sys.argv = [str(DS / "tools" / "train.py"), *train_args]
    runpy.run_path(str(DS / "tools" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
