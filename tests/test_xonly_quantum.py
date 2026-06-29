"""Quantum validation of the inversion-free x-only primitives and ladder under
boolean_simulation, on the smallest QDay curve (p=13, y^2=x^3+7, G=(11,5)).

Requires qrisp >= 0.9.5 (see the version note in
src/quantum/ec_arithmetic_xonly.py).  These run boolean_simulation and are slow
to JIT-compile, so they are kept to small basis-state cases.
"""
import pytest

qrisp = pytest.importorskip("qrisp")
from qrisp import QuantumModulus, QuantumBool, boolean_simulation, measure

import src.classical.ec_arithmetic as cl
import src.quantum.ec_arithmetic_xonly as xq

P, A, B = 13, 0, 7
G = (11, 5)
CURVE = cl.EllCurve(a=A, b=B, p=P)


def _scalar(k):
    R = cl.ell_mult_add(cl.EllPoint(*G), cl.EllZero, k, CURVE)
    return (R.x, R.y)


def _deaffinify(Xv, Zv):
    return (Xv * pow(Zv, -1, P)) % P if Zv else None


TWO_GX = _scalar(2)[0]


def test_q_xDBL():
    @boolean_simulation
    def run():
        X1 = QuantumModulus(P); X1[:] = G[0]
        Z1 = QuantumModulus(P); Z1[:] = 1
        X2, Z2 = xq.q_xDBL(X1, Z1, A, B, P)
        return measure(X2), measure(Z2)

    Xv, Zv = (int(v) for v in run())
    assert _deaffinify(Xv, Zv) == _scalar(2)[0]


def test_q_xADD():
    @boolean_simulation
    def run():
        X1 = QuantumModulus(P); X1[:] = _scalar(2)[0]   # 2G
        Z1 = QuantumModulus(P); Z1[:] = 1
        X2 = QuantumModulus(P); X2[:] = G[0]            # G ; difference = G
        Z2 = QuantumModulus(P); Z2[:] = 1
        Xp, Zp = xq.q_xADD(X1, Z1, X2, Z2, G[0], A, B, P)
        return measure(Xp), measure(Zp)

    Xv, Zv = (int(v) for v in run())
    assert _deaffinify(Xv, Zv) == _scalar(3)[0]


@pytest.mark.parametrize("k", [2, 3, 5])
def test_q_ladder_classical_scalar(k):
    @boolean_simulation
    def run():
        X0, Z0 = xq.q_ladder_x(k, G[0], TWO_GX, A, B, P)
        return measure(X0), measure(Z0)

    Xv, Zv = (int(v) for v in run())
    assert _deaffinify(Xv, Zv) == _scalar(k)[0]


@pytest.mark.parametrize("k", [2, 3])  # n=2, MSB=1 -> one cswap iteration
def test_q_ladder_quantum_scalar_cswap(k):
    """Real Shor form: scalar bits drive controlled-swaps of the ladder points."""
    n = 2
    low_bits = [(k >> i) & 1 for i in range(n - 2, -1, -1)]

    @boolean_simulation
    def run():
        kbits = []
        for bval in low_bits:
            qb = QuantumBool()
            if bval:
                qb[:] = True
            kbits.append(qb)
        X0, Z0 = xq.q_ladder_quantum(kbits, G[0], TWO_GX, A, B, P)
        return measure(X0), measure(Z0)

    Xv, Zv = (int(v) for v in run())
    assert _deaffinify(Xv, Zv) == _scalar(k)[0]
