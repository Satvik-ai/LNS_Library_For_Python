"""
ops.py
------
Arithmetic performed directly in the logarithmic domain.

* Multiplication  -> addition of log-magnitudes (exact, integer add).
* Addition/subtraction -> the classical LNS "Gaussian logarithm" identity:

      log2(x + y) = log2(x) + log2(1 + 2**r)   if sign(x) == sign(y)
      log2(x - y) = log2(x) + log2(1 - 2**r)   if sign(x) != sign(y)

  where r = log2(|y|) - log2(|x|) <= 0 (x chosen as the larger-magnitude
  operand). Crucially, this only ever consumes the two *log* values
  (la, lb) that are already stored in the LNS codes -- the linear values
  of the operands are never formed or added.
* MAC(a, b, acc) = add(multiply(a, b), acc), chained in the log domain.
"""

import math

from .core import LNSNumber, _clamp_and_flag
from .format import LNSFormat


def lns_multiply(a: LNSNumber, b: LNSNumber) -> LNSNumber:
    """LNS multiplication: log(a*b) = log(a) + log(b)."""
    assert a.fmt is b.fmt, "operands must share the same LNS format"
    fmt = a.fmt

    if a.is_zero or b.is_zero:
        return LNSNumber(0, fmt.code_min, fmt, is_zero=True)

    sign = a.sign ^ b.sign
    raw_code = a.code + b.code   # pure integer addition of fixed-point logs

    code, overflow, underflow = _clamp_and_flag(raw_code, fmt)
    if code is None:
        return LNSNumber(sign, fmt.code_min, fmt, is_zero=True, underflow=True)

    overflow = overflow or a.overflow or b.overflow
    return LNSNumber(sign, code, fmt, overflow=overflow)


def _gaussian_log_step(l_max: float, l_min: float, same_sign: bool):
    """Return log2(|result|) for two operands with log-magnitudes l_max >= l_min.
    Returns None for exact cancellation (opposite sign, equal magnitude)."""
    r = l_min - l_max  # r <= 0
    if same_sign:
        return l_max + math.log2(1.0 + 2.0 ** r)     # "sb" function
    if r == 0.0:
        return None                                   # a - a == 0 exactly
    return l_max + math.log2(1.0 - 2.0 ** r)          # "db" function


def lns_add(a: LNSNumber, b: LNSNumber) -> LNSNumber:
    """LNS addition (also handles subtraction via the sign bits),
    performed entirely on the stored log-magnitude codes."""
    assert a.fmt is b.fmt, "operands must share the same LNS format"
    fmt = a.fmt

    if a.is_zero:
        return b
    if b.is_zero:
        return a

    la = a.code / fmt.scale
    lb = b.code / fmt.scale

    if la >= lb:
        l_max, l_min, dominant_sign = la, lb, a.sign
    else:
        l_max, l_min, dominant_sign = lb, la, b.sign

    same_sign = (a.sign == b.sign)
    result_log = _gaussian_log_step(l_max, l_min, same_sign)

    if result_log is None:
        return LNSNumber(0, fmt.code_min, fmt, is_zero=True)

    raw_code = round(result_log * fmt.scale)
    code, overflow, underflow = _clamp_and_flag(raw_code, fmt)
    if code is None:
        return LNSNumber(dominant_sign, fmt.code_min, fmt, is_zero=True, underflow=True)

    overflow = overflow or a.overflow or b.overflow
    return LNSNumber(dominant_sign, code, fmt, overflow=overflow)


def lns_negate(a: LNSNumber) -> LNSNumber:
    """Unary negation: flips the sign, magnitude and zero-ness are untouched.
    Needed constantly in DNN code (gradient descent updates, residuals,
    turning a-b into a+(-b))."""
    if a.is_zero:
        return a
    return LNSNumber(1 - a.sign, a.code, a.fmt, overflow=a.overflow)


def lns_subtract(a: LNSNumber, b: LNSNumber) -> LNSNumber:
    """a - b, implemented as a + (-b) so it reuses the same Gaussian-log
    addition path -- no separate log-domain derivation needed since
    lns_add is already sign-aware."""
    return lns_add(a, lns_negate(b))


def lns_divide(a: LNSNumber, b: LNSNumber) -> LNSNumber:
    """LNS division: log(a/b) = log(a) - log(b)."""
    assert a.fmt is b.fmt, "operands must share the same LNS format"
    fmt = a.fmt

    if b.is_zero:
        # Division by zero: no finite representation -- saturate to the
        # largest representable magnitude (flagged), mirroring how
        # overflow is handled elsewhere rather than raising.
        sign = a.sign  # sign of a; b's sign is meaningless when |b|=0
        return LNSNumber(sign, fmt.code_max, fmt, overflow=True)

    if a.is_zero:
        return LNSNumber(0, fmt.code_min, fmt, is_zero=True)

    sign = a.sign ^ b.sign
    raw_code = a.code - b.code

    code, overflow, underflow = _clamp_and_flag(raw_code, fmt)
    if code is None:
        return LNSNumber(sign, fmt.code_min, fmt, is_zero=True, underflow=True)

    overflow = overflow or a.overflow or b.overflow
    return LNSNumber(sign, code, fmt, overflow=overflow)


def lns_mac(a: LNSNumber, b: LNSNumber, acc: LNSNumber) -> LNSNumber:
    """Multiply-accumulate: acc + a*b, fully in the log domain."""
    product = lns_multiply(a, b)
    return lns_add(product, acc)
