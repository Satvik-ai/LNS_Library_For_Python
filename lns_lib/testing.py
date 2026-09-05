"""
testing.py
----------
Test-vector generation and the evaluation harness that produces the
accuracy report: mean/max relative error of LNS8 and LNS16 against 
FP32 and FP16 references, for conversion, addition, multiplication 
and MAC.
"""

import random
import numpy as np

from .format import LNS16, LNS8, LNSFormat
from .core import float_to_lns, lns_to_float
from .ops import lns_add, lns_subtract, lns_multiply, lns_divide, lns_mac
from .errors import error_stats


def generate_values(n_random: int = 200, seed: int = 0):
    """Positive, negative, zero, small, large and random values."""
    values = [0.0, -0.0, 1.0, -1.0]
    values += [1e-6, -1e-6, 1e-4, -3.7e-5]          # small
    values += [1e5, -2.5e6, 9.9e7, -1e8]            # large
    rng = np.random.default_rng(seed)
    values += list(rng.uniform(-100, 100, size=n_random // 2))            # uniform random
    signs = rng.choice([-1.0, 1.0], size=n_random // 2)
    mags = 10 ** rng.uniform(-6, 4, size=n_random // 2)
    values += list(signs * mags)                                          # log-uniform random
    return values


def _finite(*vals) -> bool:
    """True if every value is finite (i.e. did not overflow to +/-inf or
    NaN when cast into the reference dtype, e.g. FP16's ~6.5e4 max)."""
    return all(np.isfinite(v) for v in vals)


def evaluate_conversion(fmt: LNSFormat, values, ref_dtype=np.float32) -> dict:
    refs, recon, skipped = [], [], 0
    for v in values:
        with np.errstate(over="ignore"):
            ref = ref_dtype(v)
        if not _finite(ref):
            # Value is outside the *reference* format's own range (e.g. a
            # 1e8 test vector cast to FP16 overflows to inf) -- not
            # comparable, and reported separately rather than as NaN.
            skipped += 1
            continue
        back = lns_to_float(float_to_lns(ref, fmt))
        refs.append(float(ref))
        recon.append(back)
    stats = error_stats(refs, recon)
    stats["skipped_out_of_ref_range"] = skipped
    return stats


def evaluate_op(fmt: LNSFormat, values, op_name: str, ref_dtype=np.float32) -> dict:
    rng = random.Random(1)
    refs, recon, skipped = [], [], 0
    for _ in range(len(values)):
        x, y = rng.choice(values), rng.choice(values)
        with np.errstate(over="ignore"):
            rx, ry = ref_dtype(x), ref_dtype(y)
        if not _finite(rx, ry):
            skipped += 1
            continue
        lx, ly = float_to_lns(rx, fmt), float_to_lns(ry, fmt)

        if op_name == "add":
            ref = float(rx) + float(ry)
            result = lns_add(lx, ly)
        elif op_name == "sub":
            ref = float(rx) - float(ry)
            result = lns_subtract(lx, ly)
        elif op_name == "mul":
            ref = float(rx) * float(ry)
            result = lns_multiply(lx, ly)
        elif op_name == "div":
            if float(ry) == 0.0:
                skipped += 1
                continue
            ref = float(rx) / float(ry)
            result = lns_divide(lx, ly)
        elif op_name == "mac":
            with np.errstate(over="ignore"):
                rz = ref_dtype(rng.choice(values))
            if not _finite(rz):
                skipped += 1
                continue
            lz = float_to_lns(rz, fmt)
            ref = float(rx) * float(ry) + float(rz)
            result = lns_mac(lx, ly, lz)
        else:
            raise ValueError(op_name)

        if not _finite(ref):
            skipped += 1
            continue

        refs.append(ref)
        recon.append(lns_to_float(result))
    stats = error_stats(refs, recon)
    stats["skipped_out_of_ref_range"] = skipped
    return stats


def full_report(n_random: int = 200) -> dict:
    values = generate_values(n_random=n_random)
    report = {}
    for fmt in (LNS16, LNS8):
        report[fmt.name] = {
            "conversion_vs_fp32": evaluate_conversion(fmt, values, np.float32),
            "conversion_vs_fp16": evaluate_conversion(fmt, values, np.float16),
            "add": evaluate_op(fmt, values, "add"),
            "mul": evaluate_op(fmt, values, "mul"),
            "mac": evaluate_op(fmt, values, "mac"),
            "sub": evaluate_op(fmt, values, "sub"),
            "div": evaluate_op(fmt, values, "div"),
            
        }
    return report


def generate_in_range_values(fmt: LNSFormat, n: int = 200, seed: int = 5):
    """Values whose magnitude -- and whose *product* with another such
    value -- stays comfortably inside fmt's representable range. Used to
    isolate the format's rounding/precision error from its range
    (overflow/underflow) limitations."""
    rng = np.random.default_rng(seed)
    hi_log = fmt.max_log2 / 2 - 2
    lo_log = fmt.min_log2 / 2 + 2
    signs = rng.choice([-1.0, 1.0], size=n)
    logs = rng.uniform(lo_log, hi_log, size=n)
    return [0.0] + list(signs * (2.0 ** logs))


def precision_report(n: int = 200) -> dict:
    """Same metrics as full_report(), but restricted to an in-range value
    set so the numbers reflect quantization/rounding precision rather than
    dynamic-range clipping. Complements full_report() for the discussion
    of "range vs precision" required by the assignment."""
    report = {}
    for fmt in (LNS16, LNS8):
        values = generate_in_range_values(fmt, n=n)
        report[fmt.name] = {
            "conversion": evaluate_conversion(fmt, values, np.float32),
            "add": evaluate_op(fmt, values, "add"),
            "mul": evaluate_op(fmt, values, "mul"),
            "mac": evaluate_op(fmt, values, "mac"),
            "sub": evaluate_op(fmt, values, "sub"),
            "div": evaluate_op(fmt, values, "div"),
            
        }
    return report


def print_report(report: dict = None) -> dict:
    if report is None:
        report = full_report()
    for fmt_name, res in report.items():
        print(f"\n=== {fmt_name} ===")
        for op_name, stats in res.items():
            skip = stats.get("skipped_out_of_ref_range", 0)
            skip_s = f"  (skipped {skip} out-of-ref-range)" if skip else ""
            print(f"  {op_name:20s}: mean_err={stats['mean']:.6e}  max_err={stats['max']:.6e}  (n={stats['n']}){skip_s}")
    return report
