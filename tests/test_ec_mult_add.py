import qrisp
import pytest
import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm
import warnings
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger


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

    result = qECarithm.q_ec_mult_add(list(P), [res_0, res_1], k, CURVE)

    meas = qrisp.multi_measurement([result[0], result[1]])
    assert meas == {(expected.x, expected.y): 1}, (
        f"k={k_val}, n_bits={n_bits}: expected ({expected.x}, {expected.y}), got {meas}"
    )


def test_ell_mult_add_dynamic():
    """Test Q + k*P under @boolean_simulation.

    q_ec_mult_add currently expects the QuantumFloat input to be fixed at trace
    time, so we keep the cases grouped in one pytest test but still trace a
    separate boolean-simulation function per k-value.
    """
    p = CURVE.p
    cases = [
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 2),
    ]

    for k_val, n_bits in cases:
        expected = clECarithm.ell_mult_add(
            clECarithm.EllPoint(*P), clECarithm.EllPoint(*Q), k_val, CURVE
        )

        @qrisp.boolean_simulation
        def run_mult_add():
            res_0 = qrisp.QuantumModulus(p)
            res_0[:] = Q[0]
            res_1 = qrisp.QuantumModulus(p)
            res_1[:] = Q[1]

            k = qrisp.QuantumFloat(n_bits)
            k[:] = k_val

            result = qECarithm.q_ec_mult_add(list(P), [res_0, res_1], k, CURVE, n_bits=n_bits)
            return qrisp.measure(result[0]), qrisp.measure(result[1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rx, ry = run_mult_add()

        faulty = [w for w in caught if "Faulty" in str(w.message)]
        assert len(faulty) == 0, (
            f"Faulty uncomputation warnings for k={k_val}, n_bits={n_bits}: "
            f"{[str(w.message) for w in faulty]}"
        )
        assert (int(rx), int(ry)) == (expected.x, expected.y), (
            f"k={k_val}, n_bits={n_bits}: expected ({expected.x}, {expected.y}), "
            f"got ({int(rx)}, {int(ry)})"
        )


def test_ell_mult_add_k_superposition():
    """Put k in an equal superposition of |1> and |2> and verify both outcomes."""
    p = CURVE.p

    expected_1 = clECarithm.ell_mult_add(
        clECarithm.EllPoint(*P), clECarithm.EllPoint(*Q), 1, CURVE
    )
    expected_2 = clECarithm.ell_mult_add(
        clECarithm.EllPoint(*P), clECarithm.EllPoint(*Q), 2, CURVE
    )

    res_0 = qrisp.QuantumModulus(p)
    res_0[:] = Q[0]
    res_1 = qrisp.QuantumModulus(p)
    res_1[:] = Q[1]

    k = qrisp.QuantumFloat(2)
    k[:] = {1: 1, 2: 1}

    result = qECarithm.q_ec_mult_add(list(P), [res_0, res_1], k, CURVE)

    meas = qrisp.multi_measurement([result[0], result[1]])
    expected_probability = 0.5
    assert (expected_1.x, expected_1.y) in meas, (
        f"Missing Q+1*P=({expected_1.x},{expected_1.y}), got {meas}"
    )
    assert (expected_2.x, expected_2.y) in meas, (
        f"Missing Q+2*P=({expected_2.x},{expected_2.y}), got {meas}"
    )
    assert meas[(expected_1.x, expected_1.y)] == pytest.approx(expected_probability, abs=1e-6), (
        f"Unexpected probability for Q+1*P: got {meas[(expected_1.x, expected_1.y)]}, "
        f"expected {expected_probability}"
    )
    assert meas[(expected_2.x, expected_2.y)] == pytest.approx(expected_probability, abs=1e-6), (
        f"Unexpected probability for Q+2*P: got {meas[(expected_2.x, expected_2.y)]}, "
        f"expected {expected_probability}"
    )
    assert len(meas) == 2, f"Expected 2 outcomes, got {len(meas)}: {meas}"


# ---- BigInteger tests ----
# Same curve y² = x³ + 5x + 4 mod 7 but with BigInteger modulus (1 limb).
BI_SIZE = 1  # 1 limb = 32 bits, enough for p=7
CURVE_BI = clECarithm.EllCurve(a=5, b=4, p=BigInteger.create_static(7, BI_SIZE))


def test_ell_mult_add_biginteger_dynamic():
    """Test Q + k*P under @boolean_simulation with BigInteger modulus."""
    p_bi = CURVE_BI.p
    cases = [
        (1, 1),
        (2, 2),
        (3, 2),
    ]

    for k_val, n_bits in cases:
        expected = clECarithm.ell_mult_add(
            clECarithm.EllPoint(*P), clECarithm.EllPoint(*Q), k_val, CURVE
        )

        @qrisp.boolean_simulation
        def run_mult_add():
            res_0 = qrisp.QuantumModulus(p_bi)
            res_0[:] = BigInteger.create_static(Q[0], BI_SIZE)
            res_1 = qrisp.QuantumModulus(p_bi)
            res_1[:] = BigInteger.create_static(Q[1], BI_SIZE)

            k = qrisp.QuantumFloat(n_bits)
            k[:] = k_val

            result = qECarithm.q_ec_mult_add(
                list(P), [res_0, res_1], k, CURVE_BI, n_bits=n_bits
            )
            return qrisp.measure(result[0]), qrisp.measure(result[1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rx, ry = run_mult_add()

        faulty = [w for w in caught if "Faulty" in str(w.message)]
        assert len(faulty) == 0, (
            f"BigInteger k={k_val}, n_bits={n_bits}: "
            f"Faulty uncomputation warnings: {[str(w.message) for w in faulty]}"
        )
        rx_int = int(rx()) if isinstance(rx, BigInteger) else int(rx)
        ry_int = int(ry()) if isinstance(ry, BigInteger) else int(ry)
        assert (rx_int, ry_int) == (expected.x, expected.y), (
            f"BigInteger k={k_val}, n_bits={n_bits}: "
            f"expected ({expected.x}, {expected.y}), got ({rx_int}, {ry_int})"
        )
