#!/usr/bin/env python3
"""Quick test for boolean_simulation paths (int + BigInteger)."""
import sys, warnings
sys.path.insert(0, '.')

import qrisp
import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

CURVE = clECarithm.EllCurve(a=5, b=4, p=7)
P = (3, 5)
Q = (0, 2)

# Test 1: boolean_simulation, int modulus, k=1, n_bits=1
print("=== Test 1: boolean_simulation int, k=1 ===")
expected = clECarithm.ell_mult_add(clECarithm.EllPoint(*P), clECarithm.EllPoint(*Q), 1, CURVE)

@qrisp.boolean_simulation
def run_int():
    res_0 = qrisp.QuantumModulus(7)
    res_0[:] = Q[0]
    res_1 = qrisp.QuantumModulus(7)
    res_1[:] = Q[1]
    k = qrisp.QuantumFloat(1)
    k[:] = 1
    result = qECarithm.q_ec_mult_add(list(P), [res_0, res_1], k, CURVE, n_bits=1)
    return qrisp.measure(result[0]), qrisp.measure(result[1])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rx, ry = run_int()
faulty = [w for w in caught if "Faulty" in str(w.message)]
print(f"Expected: ({expected.x}, {expected.y}), Got: ({int(rx)}, {int(ry)})")
assert len(faulty) == 0, f"Faulty: {[str(w.message) for w in faulty]}"
assert (int(rx), int(ry)) == (expected.x, expected.y)
print("PASS\n")

# Test 2: boolean_simulation, BigInteger modulus, k=1, n_bits=1
print("=== Test 2: boolean_simulation BigInteger, k=1 ===")
BI_SIZE = 1
p_bi = BigInteger.create_static(7, BI_SIZE)
CURVE_BI = clECarithm.EllCurve(a=5, b=4, p=p_bi)

@qrisp.boolean_simulation
def run_bi():
    res_0 = qrisp.QuantumModulus(p_bi)
    res_0[:] = BigInteger.create_static(Q[0], BI_SIZE)
    res_1 = qrisp.QuantumModulus(p_bi)
    res_1[:] = BigInteger.create_static(Q[1], BI_SIZE)
    k = qrisp.QuantumFloat(1)
    k[:] = 1
    result = qECarithm.q_ec_mult_add(list(P), [res_0, res_1], k, CURVE_BI, n_bits=1)
    return qrisp.measure(result[0]), qrisp.measure(result[1])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rx, ry = run_bi()
faulty = [w for w in caught if "Faulty" in str(w.message)]
rx_int = int(rx()) if isinstance(rx, BigInteger) else int(rx)
ry_int = int(ry()) if isinstance(ry, BigInteger) else int(ry)
print(f"Expected: ({expected.x}, {expected.y}), Got: ({rx_int}, {ry_int})")
assert len(faulty) == 0, f"Faulty: {[str(w.message) for w in faulty]}"
assert (rx_int, ry_int) == (expected.x, expected.y)
print("PASS\n")

print("=== All boolean_simulation tests passed! ===")
