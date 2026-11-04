"""Exhaustive check of the standard-form product used by the point addition.

``standard_form_product`` undoes the Montgomery factor of Qrisp's quantum-quantum
product using ``montgomery_reduction_shift``. If that shift drifts from the one
Qrisp actually applies, every product is off by a constant power of two, which is
how the dynamic EC addition tests broke with an earlier Qrisp commit. These tests
are cheap (boolean simulation, small primes) and catch that immediately.
"""

import pytest
import qrisp
from qrisp import QuantumModulus, cx, measure

import src.quantum.ec_arithmetic as qECarithm


# 17 is a Fermat prime, so (p - 1)**2 is a power of two: the edge case where
# ceil(log2((p - 1)**2 + 1)) and ceil(log2((p - 1)**2)) + 1 differ.
@pytest.mark.parametrize("p", [7, 13, 17])
def test_standard_form_product_exhaustive(p, no_faulty_uncomputation):
    @qrisp.boolean_simulation
    def product(a_val, b_val):
        a = QuantumModulus(p)
        a[:] = a_val
        b = QuantumModulus(p)
        b[:] = b_val
        out = QuantumModulus(p)
        with qECarithm.standard_form_product(a, b, p, p) as t:
            cx(t, out)
        return measure(out), measure(a), measure(b)

    wrong = []
    with no_faulty_uncomputation(f"p={p}"):
        for a_val in range(p):
            for b_val in range(p):
                out, a_out, b_out = (int(v) for v in product(a_val, b_val))
                if (out, a_out, b_out) != (a_val * b_val % p, a_val, b_val):
                    wrong.append(((a_val, b_val), (out, a_out, b_out)))

    assert not wrong, (
        f"p={p}: wrong (a, b) -> (a*b, a, b) for {len(wrong)} inputs, e.g. {wrong[:3]}"
    )


@pytest.mark.parametrize("p", [3, 5, 7, 11, 13, 17, 257, 65537, 2**255 - 19, 2**256 - 2**32 - 977])
def test_montgomery_reduction_shift_formula(p):
    """The shift equals ceil(log2((p - 1)**2)) + 1 - ceil(log2(p)), computed exactly."""

    def ceil_log2(x):
        # exact: smallest k with 2**k >= x
        k = 0
        while (1 << k) < x:
            k += 1
        return k

    assert qECarithm.montgomery_reduction_shift(p) == ceil_log2((p - 1) ** 2) + 1 - ceil_log2(p)
