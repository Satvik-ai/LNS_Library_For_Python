"""
core.py
-------
LNSNumber representation, plus conversion to/from FP32 and FP16.

Conversion is the only place floating point is touched intentionally
(it is the definition of the format). All *arithmetic* (see ops.py)
works purely on the stored (sign, code) integer fields.
"""

import math
import numpy as np

from .format import LNSFormat


class LNSNumber:
    """A single value in a given LNSFormat.

    Attributes
    ----------
    sign : int        0 = positive, 1 = negative
    code : int        two's-complement fixed point log2(|x|) code
    fmt  : LNSFormat  the format this number belongs to
    is_zero, overflow, underflow : bool  status flags set during conversion
                                          or arithmetic, useful for testing.
    """

    __slots__ = ("sign", "code", "fmt", "is_zero", "overflow", "underflow")

    def __init__(self, sign, code, fmt, is_zero=False, overflow=False, underflow=False):
        self.sign = sign
        self.code = code
        self.fmt = fmt
        self.is_zero = is_zero
        self.overflow = overflow
        self.underflow = underflow

    def __repr__(self):
        if self.is_zero:
            base = f"LNSNumber({self.fmt.name}, 0.0)"
        else:
            base = f"LNSNumber({self.fmt.name}, sign={self.sign}, code={self.code})"
        flags = "/".join(f for f, v in (("OVF", self.overflow), ("UNF", self.underflow)) if v)
        return f"{base} [{flags}]" if flags else base

    def bits(self) -> str:
        """Raw bit pattern, MSB first: sign bit followed by the two's
        complement magnitude field."""
        mag_bits = self.fmt.mag_bits
        raw = self.code & ((1 << mag_bits) - 1)
        return f"{self.sign}{raw:0{mag_bits}b}"


# ---------------------------------------------------------------------------
# Internal helper: range checking / saturation / flush-to-zero
# ---------------------------------------------------------------------------

def _clamp_and_flag(code: int, fmt: LNSFormat):
    """Clamp a raw (rounded) log-code into the representable range.

    Returns (code_or_None, overflow, underflow).
    code is None if the value underflowed and must be flushed to zero.
    """
    overflow = False
    underflow = False
    if code > fmt.code_max:
        code = fmt.code_max
        overflow = True
    elif code < fmt.code_min_nonzero:
        underflow = True
        return None, overflow, underflow
    return code, overflow, underflow


# ---------------------------------------------------------------------------
# float <-> LNS
# ---------------------------------------------------------------------------

def float_to_lns(x, fmt: LNSFormat) -> LNSNumber:
    """Convert a Python float / numpy scalar into an LNSNumber of format fmt.

    Handles: exact zero, subnormal-ish tiny values (-> underflow/flush to
    zero), values beyond range (-> overflow/saturate), and rounds to the
    nearest representable point on the log grid otherwise.
    """
    x = float(x)

    if x == 0.0:
        return LNSNumber(0, fmt.code_min, fmt, is_zero=True)

    if math.isnan(x):
        return LNSNumber(0, fmt.code_min, fmt, is_zero=True, overflow=True)

    if math.isinf(x):
        sign = 1 if x < 0 else 0
        return LNSNumber(sign, fmt.code_max, fmt, overflow=True)

    sign = 1 if x < 0 else 0
    mag = abs(x)
    log_val = math.log2(mag)
    raw_code = round(log_val * fmt.scale)          # <-- rounding to the LNS grid

    code, overflow, underflow = _clamp_and_flag(raw_code, fmt)
    if code is None:
        return LNSNumber(sign, fmt.code_min, fmt, is_zero=True, underflow=True)

    return LNSNumber(sign, code, fmt, overflow=overflow)


def lns_to_float(a: LNSNumber) -> float:
    """Convert an LNSNumber back to a (double precision) Python float.
    Double is used only as the 'infinite precision' host type for reporting;
    it plays no part in the LNS arithmetic itself."""
    if a.is_zero:
        return 0.0
    log_val = a.code / a.fmt.scale
    mag = 2.0 ** log_val
    return -mag if a.sign else mag


# Convenience wrappers that make the FP32 / FP16 source format explicit -----

def fp32_to_lns(x, fmt: LNSFormat) -> LNSNumber:
    return float_to_lns(np.float32(x), fmt)


def fp16_to_lns(x, fmt: LNSFormat) -> LNSNumber:
    return float_to_lns(np.float16(x), fmt)


def lns_to_fp32(a: LNSNumber) -> np.float32:
    return np.float32(lns_to_float(a))


def lns_to_fp16(a: LNSNumber) -> np.float16:
    return np.float16(lns_to_float(a))
