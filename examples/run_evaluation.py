"""
run_evaluation.py
------------------
Produces the accuracy report:

  - Converts positive/negative/zero/small/large/random values to LNS16 and
    LNS8 and back, comparing against FP32 and FP16 references.
  - Runs addition, multiplication and MAC in both formats and compares
    against the FP32/FP16 reference results.
  - Reports mean and max relative error (absolute error when the
    reference is exactly zero) for every case.
  - Separately reports "precision-only" error (operands/results kept
    inside each format's own dynamic range) so range-clipping effects
    can be told apart from rounding/precision effects.
  - Prints each format's numeric range and grid resolution alongside
    FP32/FP16 for direct comparison.

Run with:  python3 examples/run_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from lns_lib import LNS16, LNS8, print_report, precision_report


def print_precision_report(report):
    for fmt_name, res in report.items():
        print(f"\n=== {fmt_name} (operands & results kept in-range) ===")
        for op_name, stats in res.items():
            print(f"  {op_name:12s}: mean_err={stats['mean']:.6e}  max_err={stats['max']:.6e}  (n={stats['n']})")


def print_format_summary():
    print("\n=== Numeric range / precision summary ===")
    for fmt in (LNS16, LNS8):
        print(
            f"{fmt.name:6s} ({fmt.total_bits} bits = 1 sign + {fmt.int_bits} int + "
            f"{fmt.frac_bits} frac): "
            f"max|x|={fmt.max_abs_value:.4g}  min nonzero |x|={fmt.min_abs_value:.4g}  "
            f"grid step={fmt.quantum_relative_step * 100:.3f}% (relative)"
        )
    f32 = np.finfo(np.float32)
    f16 = np.finfo(np.float16)
    print(f"FP32  (32 bits): max={f32.max:.4g}  min normal={f32.tiny:.4g}  eps={f32.eps:.4g}")
    print(f"FP16  (16 bits): max={f16.max:.4g}  min normal={f16.tiny:.4g}  eps={f16.eps:.4g}")


if __name__ == "__main__":
    print("############ FULL RANDOM TEST SET (spans 1e-6 .. 1e8) ############")
    full = print_report()

    print("\n############ PRECISION-ONLY TEST SET (in-range only) ############")
    prec = precision_report()
    print_precision_report(prec)

    print_format_summary()
