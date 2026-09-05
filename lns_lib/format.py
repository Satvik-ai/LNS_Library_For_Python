"""
format.py
---------
Defines the bit-field layout of a Logarithmic Number System (LNS) format.

An LNS number is stored as:
    [ sign bit | signed fixed-point log2(|x|), two's complement ]

The fixed-point log field has `int_bits` integer bits and `frac_bits`
fractional bits (a Q(int_bits).(frac_bits) format). The most negative
two's-complement code in that field is *reserved* to represent exact
zero, since log2(0) = -infinity cannot itself be stored.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LNSFormat:
    name: str
    total_bits: int   # total bits including the sign bit
    int_bits: int     # integer bits of the log-magnitude field
    frac_bits: int    # fractional bits of the log-magnitude field

    def __post_init__(self):
        assert 1 + self.int_bits + self.frac_bits == self.total_bits, (
            f"{self.name}: 1 (sign) + int_bits + frac_bits must equal total_bits "
            f"(got 1 + {self.int_bits} + {self.frac_bits} != {self.total_bits})"
        )

    @property
    def mag_bits(self) -> int:
        """Bits used for the two's-complement log-magnitude field."""
        return self.int_bits + self.frac_bits

    @property
    def scale(self) -> int:
        """Number of fixed-point steps per unit of log2 magnitude (2**frac_bits)."""
        return 1 << self.frac_bits

    @property
    def code_max(self) -> int:
        """Largest usable two's-complement magnitude code."""
        return (1 << (self.mag_bits - 1)) - 1

    @property
    def code_min(self) -> int:
        """Most negative two's-complement code -- RESERVED to mean 'zero'."""
        return -(1 << (self.mag_bits - 1))

    @property
    def code_min_nonzero(self) -> int:
        """Smallest magnitude code that represents a genuine (non-zero) value."""
        return self.code_min + 1

    @property
    def max_log2(self) -> float:
        return self.code_max / self.scale

    @property
    def min_log2(self) -> float:
        return self.code_min_nonzero / self.scale

    @property
    def max_abs_value(self) -> float:
        return 2.0 ** self.max_log2

    @property
    def min_abs_value(self) -> float:
        """Smallest representable non-zero magnitude (before flush-to-zero)."""
        return 2.0 ** self.min_log2

    @property
    def quantum_relative_step(self) -> float:
        """Approx. worst-case relative rounding step from the log grid:
        two neighbouring codes differ by 2**(1/scale) in ratio."""
        return 2.0 ** (1.0 / self.scale) - 1.0


# ---------------------------------------------------------------------------
# Presets required by the assignment
# ---------------------------------------------------------------------------

# LNS16: 1 sign + 7 integer bits + 8 fractional bits  (Q7.8 log field)
LNS16 = LNSFormat(name="LNS16", total_bits=16, int_bits=7, frac_bits=8)

# LNS8: 1 sign + 4 integer bits + 3 fractional bits   (Q4.3 log field)
LNS8 = LNSFormat(name="LNS8", total_bits=8, int_bits=4, frac_bits=3)
