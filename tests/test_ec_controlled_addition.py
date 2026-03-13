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

@pytest.mark.skip
def test_controlled_addition_ctrl_on_boolean_sim():
    """ctrl=|1> under @boolean_simulation: addition should happen."""

    @qrisp.boolean_simulation
    def run():
        ctrl = qrisp.QuantumBool()
        ctrl[:] = True
        anc_0 = qrisp.QuantumModulus(P_BS_p)
        anc_0[:] = P_BS[0]
        anc_1 = qrisp.QuantumModulus(P_BS_p)
        anc_1[:] = P_BS[1]

        with qrisp.control(ctrl):
            qECarithm.q_ec_add_inpl([anc_0, anc_1], list(G_BS), P_BS_p)

        return qrisp.measure(anc_0), qrisp.measure(anc_1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rx, ry = run()

    faulty = [w for w in caught if "Faulty" in str(w.message)]
    assert len(faulty) == 0, (
        f"Faulty uncomputation warnings: {[str(w.message) for w in faulty]}"
    )
    assert (int(rx), int(ry)) == (R_BS.x, R_BS.y), (
        f"ctrl=1 boolean_sim: got ({int(rx)}, {int(ry)}), "
        f"expected ({R_BS.x}, {R_BS.y})"
    )

@pytest.mark.skip
def test_controlled_addition_ctrl_off_boolean_sim():
    """ctrl=|0> under @boolean_simulation: result should be P unchanged."""

    @qrisp.boolean_simulation
    def run():
        ctrl = qrisp.QuantumBool()
        ctrl[:] = False
        anc_0 = qrisp.QuantumModulus(P_BS_p)
        anc_0[:] = P_BS[0]
        anc_1 = qrisp.QuantumModulus(P_BS_p)
        anc_1[:] = P_BS[1]

        with qrisp.control(ctrl):
            qECarithm.q_ec_add_inpl([anc_0, anc_1], list(G_BS), P_BS_p)

        return qrisp.measure(anc_0), qrisp.measure(anc_1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rx, ry = run()

    faulty = [w for w in caught if "Faulty" in str(w.message)]
    assert len(faulty) == 0, (
        f"Faulty uncomputation warnings: {[str(w.message) for w in faulty]}"
    )
    assert (int(rx), int(ry)) == (P_BS[0], P_BS[1]), (
        f"ctrl=0 boolean_sim: got ({int(rx)}, {int(ry)}), "
        f"expected ({P_BS[0]}, {P_BS[1]})"
    )


def test_controlled_addition_ctrl_superposition():
    """ctrl in superposition |0>+|1>: should get both P and P+G outcomes."""
    ctrl = qrisp.QuantumBool()
    qrisp.h(ctrl)

    anc_0 = qrisp.QuantumModulus(CURVE.p)
    anc_0[:] = P[0]
    anc_1 = qrisp.QuantumModulus(CURVE.p)
    anc_1[:] = P[1]

    with qrisp.control(ctrl):
        qECarithm.q_ec_add_inpl([anc_0, anc_1], list(G), CURVE.p)

    results = qrisp.multi_measurement([ctrl, anc_0, anc_1])
    # ctrl=False → P unchanged, ctrl=True → P+G
    assert (False, P[0], P[1]) in results, (
        f"Missing ctrl=0 outcome (P unchanged): {results}"
    )
    assert (True, R.x, R.y) in results, (
        f"Missing ctrl=1 outcome (P+G): {results}"
    )
    assert len(results) == 2, f"Expected 2 outcomes, got {len(results)}: {results}"
