import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm
import qrisp
import pytest


# Curve: y² = x³ + 5x + 4 mod 7
CURVE = clECarithm.EllCurve(a=5, b=4, p=7)
P = (3, 5)  # base point (power)
Q = (0, 2)  # initial point (res)


@pytest.mark.parametrize("k_val,n_bits", [
    (1, 1),   # single iteration: Q + 1*P
    (1, 2),   # two iterations, second is no-op: Q + 1*P
    (2, 2),   # two iterations: Q + 2*P
    (3, 2),   # two iterations, both fire: Q + 3*P
])
def test_ell_mult_add(k_val, n_bits):
    """Test Q + k*P for small k values."""
    p = CURVE.p

    expected = clECarithm.ell_mult_add(
        clECarithm.EllPoint(*P), clECarithm.EllPoint(*Q), k_val, CURVE
    )

    res_0 = qrisp.QuantumModulus(p)
    res_0[:] = Q[0]
    res_1 = qrisp.QuantumModulus(p)
    res_1[:] = Q[1]

    k = qrisp.QuantumFloat(n_bits)
    k[:] = k_val

    result = qECarithm.qrisp_ell_mult_add(list(P), [res_0, res_1], k, CURVE)

    meas = qrisp.multi_measurement([result[0], result[1]])
    assert meas == {(expected.x, expected.y): 1}, (
        f"k={k_val}, n_bits={n_bits}: expected ({expected.x}, {expected.y}), got {meas}"
    )
