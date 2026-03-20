import qrisp
import pytest
import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm
import warnings


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
    """Test Q + k*P under @boolean_simulation while reusing cached circuits."""
    p = CURVE.p
    cases_by_n_bits = {
        1: [1],
        2: [1, 2, 3],
    }

    for n_bits, k_values in cases_by_n_bits.items():
        @qrisp.boolean_simulation
        def run_mult_add(k_val):
            res_0 = qrisp.QuantumModulus(p)
            res_0[:] = Q[0]
            res_1 = qrisp.QuantumModulus(p)
            res_1[:] = Q[1]

            k = qrisp.QuantumFloat(n_bits)
            k[:] = k_val

            result = qECarithm.q_ec_mult_add(list(P), [res_0, res_1], k, CURVE)
            return qrisp.measure(result[0]), qrisp.measure(result[1])

        for k_val in k_values:
            expected = clECarithm.ell_mult_add(
                clECarithm.EllPoint(*P), clECarithm.EllPoint(*Q), k_val, CURVE
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rx, ry = run_mult_add(k_val)

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
