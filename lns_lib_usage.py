from lns_lib import (
    LNS16, LNS8, float_to_lns, lns_to_float,
    lns_add, lns_subtract, lns_negate, lns_multiply, lns_divide, lns_mac,
)

a = 3.5
b = -1.25

print("=" * 60)
print("STEP 1: Original float inputs")
print("=" * 60)
print(f"a = {a}  (float)")
print(f"b = {b}  (float)")

a = float_to_lns(a, LNS16)
b = float_to_lns(b, LNS16)

print()
print("=" * 60)
print("STEP 2: Converted to LNS16 representation")
print("=" * 60)
print(f"a in LNS16 = {a}")
print(f"b in LNS16 = {b}")

print()
print("=" * 60)
print("STEP 3: Arithmetic operations in LNS domain")
print("=" * 60)

result_mul = lns_to_float(lns_multiply(a, b))
print(f"Multiplication  a * b = {result_mul}   "
      f"(expected approx -4.375; multiplication is exact in LNS since it's just log-addition)")

result_add = lns_to_float(lns_add(a, b))
print(f"Addition        a + b = {result_add}   "
      f"(expected approx 2.25; addition needs a lookup/interpolation step in LNS, so more error creeps in)")

result_sub = lns_to_float(lns_subtract(a, b))
print(f"Subtraction     a - b = {result_sub}   "
      f"(expected approx 4.75; same log-domain addition mechanism as above, applied to a - b)")

result_div = lns_to_float(lns_divide(a, b))
print(f"Division        a / b = {result_div}   "
      f"(expected approx -2.8; like multiplication, division is exact log-subtraction)")

result_neg = lns_to_float(lns_negate(a))
print(f"Negation          -a  = {result_neg}   "
      f"(negation just flips the sign bit — any deviation from -3.5 is quantization "
      f"error already baked into 'a' when it was first encoded, not new error from negation)")

print()
print("=" * 60)
print("STEP 4: Multiply-accumulate (MAC) operation")
print("=" * 60)
acc = float_to_lns(0.0, LNS16)
acc = lns_mac(a, b, acc)
print(f"acc = 0 + a*b = {lns_to_float(acc)}   "
      f"(MAC fuses a multiply and an add into one LNS op, common in DNN dot-products)")

print()
print("=" * 60)
print("STEP 5: Inspecting raw bit pattern & overflow/saturation behavior")
print("=" * 60)
x = float_to_lns(1e10, LNS8)
print(f"1e10 encoded in LNS8 -> bits = {x.bits()}, overflow flag = {x.overflow}")
print("LNS8 has a very narrow dynamic range (only 8 bits total), so a large value "
      "like 1e10 can't be represented -> it saturates to the format's max magnitude "
      "and the overflow flag is set to signal that precision/range was lost.")