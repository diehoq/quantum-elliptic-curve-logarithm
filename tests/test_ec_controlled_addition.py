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


def test_controlled_addition_ctrl_on():
    """ctrl=|1>: addition should happen, result = P + G."""
    ctrl = qrisp.QuantumBool()
    ctrl[:] = True
    anc_0 = qrisp.QuantumModulus(CURVE.p)
    anc_0[:] = P[0]
    anc_1 = qrisp.QuantumModulus(CURVE.p)
    anc_1[:] = P[1]

    with qrisp.control(ctrl):
        qECarithm.qrisp_ell_add_inpl([anc_0, anc_1], list(G), CURVE.p)

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
        qECarithm.qrisp_ell_add_inpl([anc_0, anc_1], list(G), CURVE.p)

    result = qrisp.multi_measurement([anc_0, anc_1])
    assert result == {(P[0], P[1]): 1}, (
        f"ctrl=0: expected ({P[0]}, {P[1]}), got {result}"
    )
