"""
errors.py
---------
Error metrics for comparing LNS results against FP32/FP16 references.
"""

import numpy as np


def relative_error(x_ref, x_test) -> float:
    """|x_ref - x_test| / |x_ref|, or absolute error if x_ref == 0."""
    x_ref = float(x_ref)
    x_test = float(x_test)
    if x_ref == 0.0:
        return abs(x_test - x_ref)
    return abs(x_ref - x_test) / abs(x_ref)


def error_stats(ref_values, test_values) -> dict:
    """Mean and max error over a list of (ref, test) pairs."""
    errs = np.array([relative_error(r, t) for r, t in zip(ref_values, test_values)])
    return {"mean": float(np.mean(errs)), "max": float(np.max(errs)), "n": int(len(errs))}
