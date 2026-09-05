"""
lns_lib
=======
A small, importable library implementing Logarithmic Number System (LNS)
arithmetic (LNS16 and LNS8) for use in DNN accuracy / speed experiments.
"""

from .format import LNSFormat, LNS16, LNS8
from .core import (
    LNSNumber,
    float_to_lns, lns_to_float,
    fp32_to_lns, fp16_to_lns,
    lns_to_fp32, lns_to_fp16,
)
from .ops import lns_add, lns_subtract, lns_negate, lns_multiply, lns_divide, lns_mac
from .errors import relative_error, error_stats
from .testing import generate_values, full_report, print_report, precision_report

__all__ = [
    "LNSFormat", "LNS16", "LNS8",
    "LNSNumber",
    "float_to_lns", "lns_to_float",
    "fp32_to_lns", "fp16_to_lns",
    "lns_to_fp32", "lns_to_fp16",
    "lns_add", "lns_subtract", "lns_negate", "lns_multiply", "lns_divide", "lns_mac",
    "relative_error", "error_stats",
    "generate_values", "full_report", "print_report", "precision_report",
]

__version__ = "0.1.0"
