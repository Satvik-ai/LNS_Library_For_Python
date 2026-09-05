# LNS Library for Python

A small, importable Python library implementing **Logarithmic Number System
(LNS)** arithmetic in two formats, **LNS16** and **LNS8**, intended for
studying reduced-precision arithmetic in Deep Neural Network (DNN) computation.

## 1. Number format

An LNS value is stored as `sign` + a fixed-point, two's-complement encoding
of `log2(|x|)`:

| Format | Total bits | Sign | Integer bits | Fractional bits | Layout |
|--------|-----------|------|---------------|------------------|--------|
| LNS16  | 16        | 1    | 7             | 8                | Q7.8   |
| LNS8   | 8         | 1    | 4             | 3                | Q4.3   |

The most negative two's-complement code in the magnitude field is reserved
to mean **exact zero** (since `log2(0)` is undefined). Values whose rounded
log-magnitude exceeds the representable range are **saturated** (overflow,
flagged `.overflow`); values that fall below the smallest representable
non-zero magnitude are **flushed to zero** (underflow, flagged `.underflow`).

This is a design choice, not a standard -- see `lns_lib/format.py`, where
`LNSFormat(...)` lets you define other bit layouts to explore
the range/precision trade-off further.

## 2. Why addition is the hard part

In LNS, **multiplication is addition of logs** (exact, just an integer add),
and **division is subtraction of logs** (also exact). **Addition/subtraction
are not native LNS operations** -- they require the classical
Gaussian-logarithm identity, so that everything still happens on the stored
log values, never by reconstructing the linear operands:

```
log2(x + y) = log2(x) + log2(1 + 2**r)      if sign(x) == sign(y)
log2(x - y) = log2(x) + log2(1 - 2**r)      if sign(x) != sign(y)
```

where `x` is the larger-magnitude operand and `r = log2(|y|) - log2(|x|) <= 0`.
See `_gaussian_log_step()` in `lns_lib/ops.py`.
MAC is simply `add(multiply(a, b), acc)`, chained entirely in the log domain.

## 3. Operations provided
 
All six arithmetic operations are exposed as free functions in `lns_lib/ops.py`, each
operating directly on the stored `(sign, code)` log-domain fields -- the linear value
is never reconstructed mid-computation.

* **`lns_add(a, b)`** -- addition (and, via sign flip, subtraction), performed with the
  classical Gaussian-logarithm identity described in Section 2:
  `log2(x + y) = log2(x) + log2(1 + 2**r)` (same sign) or
  `log2(x - y) = log2(x) + log2(1 - 2**r)` (opposite sign), where `r = log2(|y|) -
  log2(|x|) <= 0`. Unlike multiply/divide, this is **not exact** even ignoring input
  quantization: it evaluates a transcendental correction term, and that term is highly
  sensitive to rounding error whenever the two operands nearly cancel. Handles operands
  of either sign transparently -- exact cancellation (`a + (-a)`) returns precise zero.
* **`lns_multiply(a, b)`** -- multiplication as log-addition: `log2(a*b) = log2(a) +
  log2(b)`, computed as a single integer add of the two magnitude codes with the sign
  bits XORed. This is **exact** in the log domain (the only error is the quantization
  already present in `a` and `b` from conversion). Used for weight/activation products
  in convolution and fully-connected layers.
* **`lns_mac(a, b, acc)`** -- multiply-accumulate, defined as
  `lns_add(lns_multiply(a, b), acc)`, chained entirely in the log domain. This is the
  workhorse of DNN inference: a convolution or fully-connected layer is a long chain of
  MACs, so its accuracy is governed by the exact-multiply / approximate-add trade-off
  described above, repeated once per accumulation step.
* **`lns_divide(a, b)`** -- division as log-subtraction: `log2(a/b) = log2(a) -
  log2(b)`, computed as a single integer subtract of the magnitude codes, sign bits
  XORed. Also **exact** given the inputs' existing quantization. Division by zero
  saturates to the format's maximum magnitude with `.overflow` set rather than raising;
  zero divided by a non-zero value returns exact zero. Needed for batch/layer-norm's
  division by standard deviation and for softmax normalization.
* **`lns_subtract(a, b)`** -- implemented as `lns_add(a, lns_negate(b))`, reusing the
  same sign-aware Gaussian-log path with no separate derivation needed. Inherits the
  same near-cancellation sensitivity as `lns_add`. Used for residual connections and
  loss terms such as `(y_pred - y_true)`.
* **`lns_negate(a)`** -- unary negation. Since the stored value is `sign` +
  `log2(|x|)`, negation is just a sign-bit flip; magnitude, zero-ness, and any
  overflow/underflow flags are left untouched. Essentially free in the log domain, and
  used constantly for gradient-descent updates (`w -= lr * grad`) and turning
  `a - b` into `a + (-b)`.

**Caveat surfaced by testing `lns_divide` on LNS8:** if either operand has
already overflowed or underflowed *at conversion time* (LNS8's range is
only ~0.004 to ~235), the division operates on the saturated/zeroed
representation, not the true value -- e.g. `444 / -1e8` comes out as `-1.0`
because both operands individually saturate to the same ceiling magnitude
before the division ever runs. This isn't a division bug; it's a direct
consequence of LNS8's narrow dynamic range, and it's worth calling out
explicitly in the range-vs-precision discussion.

## 4. Package layout

```
lns_lib/
  format.py    LNSFormat dataclass + LNS16 / LNS8 presets
  core.py      LNSNumber class, float_to_lns / lns_to_float, fp32/fp16 wrappers
  ops.py       lns_add, lns_multiply, lns_mac
  errors.py    relative_error, error_stats
  testing.py   test-vector generation + full_report()/precision_report()

tests/
  test_lns.py                 pytest suite

examples/
  run_evaluation.py           produces the accuracy report
```

## 5. Install the library

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e . # or add the repo root to your `PYTHONPATH` / `sys.path`
python -c "from lns_lib import LNS16, LNS8; print(LNS16, LNS8)" # Verify the install
```

## 6. Usage

```bash
python3 lns_lib_usage.py
```

## 7. Running the tests / accuracy report

```bash
python3 -m pytest tests/                
python3 examples/run_evaluation.py      # prints the full accuracy report
```

## 8. PyTest Report

[Click Here](https://drive.google.com/file/d/1MjRqAfeodW2IJA9sBi5eZx-jIp1Fqk-i/view?usp=sharing) to see the PyTest Report.

## 9. Accuracy report (measured on this build)

`examples/run_evaluation.py` compares LNS16/LNS8 conversion, add, multiply,
MAC, subtract and division against FP32/FP16 references using zero, positive/negative,
small/large and randomly generated values (relative error, or absolute
error when the reference is exactly zero). Two views are reported:

* **Full range test** -- values spanning ~1e-6 to ~1e8, so LNS8's very
  narrow native range (~0.004 to ~235) is frequently exceeded and its
  errors include saturation/flush-to-zero effects.
* **Precision-only test** -- operands (and their products/sums) are kept
  inside each format's own dynamic range, isolating rounding/quantization
  error from range-clipping error.

[Click Here](https://drive.google.com/file/d/1lbyCAIZi4j_WudYLS3O1EqApCZf8TBjU/view?usp=sharing) to see the Accuracy report.
