import numpy as np
import pytest

from lns_lib import (
    LNS16, LNS8,
    float_to_lns, lns_to_float,
    lns_add, lns_subtract, lns_negate, lns_multiply, lns_divide, lns_mac,
    relative_error,
)


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
def test_zero_roundtrip(fmt):
    z = float_to_lns(0.0, fmt)
    assert z.is_zero
    assert lns_to_float(z) == 0.0


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
@pytest.mark.parametrize("x", [1.0, -1.0, 2.0, -0.5, 123.456, -0.0001])
def test_conversion_roundtrip_sign_and_error(fmt, x):
    v = float_to_lns(x, fmt)
    back = lns_to_float(v)
    assert (back >= 0) == (x >= 0)
    # LNS8 is coarse (3 fractional bits) -> use a loose bound
    bound = 0.3 if fmt is LNS8 else 0.02
    assert relative_error(x, back) < bound


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
def test_multiply_matches_reference(fmt):
    rng = np.random.default_rng(0)
    bound = 0.35 if fmt is LNS8 else 0.03
    for _ in range(100):
        x, y = rng.uniform(-50, 50, 2)
        if x == 0 or y == 0:
            continue
        lx, ly = float_to_lns(x, fmt), float_to_lns(y, fmt)
        result = lns_to_float(lns_multiply(lx, ly))
        ref = x * y
        assert relative_error(ref, result) < bound


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
def test_add_matches_reference(fmt):
    rng = np.random.default_rng(1)
    bound = 0.45 if fmt is LNS8 else 0.05
    for _ in range(100):
        x, y = rng.uniform(-50, 50, 2)
        lx, ly = float_to_lns(x, fmt), float_to_lns(y, fmt)
        result = lns_to_float(lns_add(lx, ly))
        ref = x + y
        if abs(ref) < 1e-3:
            assert abs(result) < 1.0
        else:
            assert relative_error(ref, result) < bound


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
def test_mac_matches_reference(fmt):
    rng = np.random.default_rng(2)
    bound = 0.5 if fmt is LNS8 else 0.06
    for _ in range(100):
        x, y, z = rng.uniform(-20, 20, 3)
        lx, ly, lz = (float_to_lns(v, fmt) for v in (x, y, z))
        result = lns_to_float(lns_mac(lx, ly, lz))
        ref = x * y + z
        if abs(ref) < 1e-3:
            assert abs(result) < 1.0
        else:
            assert relative_error(ref, result) < bound


def test_overflow_saturates():
    v = float_to_lns(1e10, LNS8)  # far beyond LNS8's range
    assert v.overflow


def test_underflow_flushes_to_zero():
    v = float_to_lns(1e-20, LNS8)  # far below LNS8's range
    assert v.is_zero


def test_multiply_by_zero_is_zero():
    for fmt in (LNS16, LNS8):
        z = float_to_lns(0.0, fmt)
        x = float_to_lns(42.0, fmt)
        assert lns_multiply(x, z).is_zero
        assert lns_multiply(z, x).is_zero


def test_exact_cancellation_is_zero():
    for fmt in (LNS16, LNS8):
        x = float_to_lns(7.5, fmt)
        neg_x = float_to_lns(-7.5, fmt)
        result = lns_add(x, neg_x)
        assert result.is_zero


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
def test_negate(fmt):
    x = float_to_lns(5.0, fmt)
    nx = lns_negate(x)
    assert lns_to_float(nx) == -lns_to_float(x)
    assert lns_negate(lns_negate(x)).sign == x.sign
    z = float_to_lns(0.0, fmt)
    assert lns_negate(z).is_zero


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
def test_subtract_matches_reference(fmt):
    rng = np.random.default_rng(3)
    bound = 0.45 if fmt is LNS8 else 0.05
    for _ in range(100):
        x, y = rng.uniform(-50, 50, 2)
        lx, ly = float_to_lns(x, fmt), float_to_lns(y, fmt)
        result = lns_to_float(lns_subtract(lx, ly))
        ref = x - y
        if abs(ref) < 1e-3:
            assert abs(result) < 1.0
        else:
            assert relative_error(ref, result) < bound


@pytest.mark.parametrize("fmt", [LNS16, LNS8])
def test_divide_matches_reference(fmt):
    rng = np.random.default_rng(4)
    bound = 0.35 if fmt is LNS8 else 0.03
    for _ in range(100):
        x, y = rng.uniform(1, 50, 2)  # keep positive & away from zero
        lx, ly = float_to_lns(x, fmt), float_to_lns(y, fmt)
        result = lns_to_float(lns_divide(lx, ly))
        ref = x / y
        assert relative_error(ref, result) < bound


def test_divide_by_zero_saturates():
    for fmt in (LNS16, LNS8):
        a = float_to_lns(5.0, fmt)
        z = float_to_lns(0.0, fmt)
        result = lns_divide(a, z)
        assert result.overflow


def test_zero_divided_by_nonzero_is_zero():
    for fmt in (LNS16, LNS8):
        z = float_to_lns(0.0, fmt)
        a = float_to_lns(5.0, fmt)
        assert lns_divide(z, a).is_zero
