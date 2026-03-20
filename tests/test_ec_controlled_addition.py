import warnings

import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm
import qrisp
import pytest


# Smoke test: p=7, y²=x³+5x+4, P=(0,2), G=(3,5)
P = (0, 2)
G = (3, 5)
CURVE = clECarithm.EllCurve(a=5, b=4, p=7)
R = clECarithm.ell_add_generic(
    clECarithm.EllPoint(*P), clECarithm.EllPoint(*G), CURVE
)

# Boolean simulation tests params:
P_BS = (0, 2)
G_BS = (3, 5)
P_BS_p = 7
R_BS = clECarithm.ell_add_generic(
    clECarithm.EllPoint(*P_BS), clECarithm.EllPoint(*G_BS),
    clECarithm.EllCurve(a=5, b=4, p=P_BS_p),
)


def test_controlled_addition_ctrl_on():
    """ctrl=|1>: addition should happen, result = P + G."""
    ctrl = qrisp.QuantumBool()
    ctrl[:] = True
    anc_0 = qrisp.QuantumModulus(CURVE.p)
    anc_0[:] = P[0]
    anc_1 = qrisp.QuantumModulus(CURVE.p)
    anc_1[:] = P[1]

    with qrisp.control(ctrl):
        qECarithm.q_ec_add_inpl([anc_0, anc_1], list(G), CURVE.p)

    result = qrisp.multi_measurement([anc_0, anc_1])
    assert result == {(R.x, R.y): 1}, (
        f"ctrl=1: expected ({R.x}, {R.y}), got {result}"
    )


def test_controlled_addition_ctrl_off():
    """ctrl=|0>: addition should NOT happen, result = P unchanged."""
    ctrl = qrisp.QuantumBool()
    ctrl[:] = False
    anc_0 = qrisp.QuantumModulus(CURVE.p)
    anc_0[:] = P[0]
    anc_1 = qrisp.QuantumModulus(CURVE.p)
    anc_1[:] = P[1]

    with qrisp.control(ctrl):
        qECarithm.q_ec_add_inpl([anc_0, anc_1], list(G), CURVE.p)

    result = qrisp.multi_measurement([anc_0, anc_1])
    assert result == {(P[0], P[1]): 1}, (
        f"ctrl=0: expected ({P[0]}, {P[1]}), got {result}"
    )

def test_controlled_addition_boolean_sim():
    """Test ctrl=|0> and ctrl=|1> under one cached @boolean_simulation circuit."""
    @qrisp.boolean_simulation
    def run(ctrl_value):
        ctrl = qrisp.QuantumBool()
        ctrl[:] = ctrl_value
        anc_0 = qrisp.QuantumModulus(P_BS_p)
        anc_0[:] = P_BS[0]
        anc_1 = qrisp.QuantumModulus(P_BS_p)
        anc_1[:] = P_BS[1]

        with qrisp.control(ctrl):
            qECarithm.q_ec_add_inpl([anc_0, anc_1], list(G_BS), P_BS_p)

        return qrisp.measure(anc_0), qrisp.measure(anc_1)

    cases = [
        (True, (R_BS.x, R_BS.y), "ctrl=1"),
        (False, (P_BS[0], P_BS[1]), "ctrl=0"),
    ]

    for ctrl_value, expected, label in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rx, ry = run(ctrl_value)

        faulty = [w for w in caught if "Faulty" in str(w.message)]
        assert len(faulty) == 0, (
            f"Faulty uncomputation warnings for {label}: "
            f"{[str(w.message) for w in faulty]}"
        )
        assert (int(rx), int(ry)) == expected, (
            f"{label} boolean_sim: got ({int(rx)}, {int(ry)}), "
            f"expected {expected}"
        )


def test_controlled_addition_ctrl_superposition():
    """Prepare ctrl in an equal superposition of |0> and |1>."""
    ctrl = qrisp.QuantumBool()
    ctrl[:] = {False: 1, True: 1}

    anc_0 = qrisp.QuantumModulus(CURVE.p)
    anc_0[:] = P[0]
    anc_1 = qrisp.QuantumModulus(CURVE.p)
    anc_1[:] = P[1]

    with qrisp.control(ctrl):
        qECarithm.q_ec_add_inpl([anc_0, anc_1], list(G), CURVE.p)

    results = qrisp.multi_measurement([ctrl, anc_0, anc_1])
    expected_probability = 0.5
    # ctrl=False → P unchanged, ctrl=True → P+G
    assert (False, P[0], P[1]) in results, (
        f"Missing ctrl=0 outcome (P unchanged): {results}"
    )
    assert (True, R.x, R.y) in results, (
        f"Missing ctrl=1 outcome (P+G): {results}"
    )
    assert results[(False, P[0], P[1])] == pytest.approx(expected_probability, abs=1e-6), (
        f"Unexpected probability for ctrl=0 branch: got {results[(False, P[0], P[1])]}, "
        f"expected {expected_probability}"
    )
    assert results[(True, R.x, R.y)] == pytest.approx(expected_probability, abs=1e-6), (
        f"Unexpected probability for ctrl=1 branch: got {results[(True, R.x, R.y)]}, "
        f"expected {expected_probability}"
    )
    assert len(results) == 2, f"Expected 2 outcomes, got {len(results)}: {results}"
