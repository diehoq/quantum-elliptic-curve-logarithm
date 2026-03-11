"""
Smoke test for Shor's algorithm for the elliptic curve discrete logarithm.

Curve: y² = x³ + 3x + 3  mod 5   (matching Shor_EC_discrete_log.ipynb)
Generator G = (3, 2), target P = (3, 3)
Group order = 5, discrete log k = 4  (4*G = P)
Initial ancilla point P0 = (3, 2)

We build the quantum circuit (without running full simulation)
and verify the classical pieces are consistent.
"""
import qrisp
import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm


def test_shor_ec_dlog_circuit_builds():
    """Verify the Shor EC discrete log circuit builds without error."""
    p = 5
    curve = clECarithm.EllCurve(a=3, b=3, p=p)

    G = [3, 2]
    P = [3, 3]

    # Classical sanity: verify discrete log exists
    k_cl = clECarithm.compute_logarithm(
        clECarithm.EllPoint(*G), clECarithm.EllPoint(*P), curve
    )
    assert k_cl is not None, f"No discrete log found for P={P} on curve"
    assert k_cl == 4, f"Expected k=4, got k={k_cl}"

    # Build the quantum circuit (matching the notebook exactly)
    n_bits = p.bit_length()
    x1 = qrisp.QuantumModulus(2**n_bits)
    x2 = qrisp.QuantumModulus(2**n_bits)

    mod_p = qrisp.QuantumModulus(p)
    anc = qrisp.QuantumArray(qtype=mod_p, shape=(2,))
    anc[:] = [3, 2]

    qrisp.h(x1)
    qrisp.h(x2)

    # Multiplication steps: anc = P0 + x1*G, then anc = anc - x2*P
    anc = qECarithm.q_ec_mult_add(G, anc, x1, curve)
    anc = qECarithm.q_ec_mult_add(P, anc, x2, curve)

    # Inverse QFT
    qrisp.QFT(x1, inv=True)
    qrisp.QFT(x2, inv=True)

    # Verify circuit was built (check qubit count is reasonable)
    num_qubits = len([qb for qb in anc[0].qs.qubits if qb.allocated])
    assert num_qubits > 0, "No qubits allocated"


def test_shor_ec_dlog_classical_consistency():
    """Verify the classical discrete log computation is consistent."""
    p = 5
    curve = clECarithm.EllCurve(a=3, b=3, p=p)

    G_pt = clECarithm.EllPoint(3, 2)
    P_pt = clECarithm.EllPoint(3, 3)

    k = clECarithm.compute_logarithm(G_pt, P_pt, curve)
    assert k is not None, "Discrete log not found"
    assert k == 4, f"Expected k=4, got k={k}"

    # Verify: k * G == P
    result = G_pt
    for _ in range(k - 1):
        result = clECarithm.ell_add(result, G_pt, curve)

    assert result.x == P_pt.x and result.y == P_pt.y, (
        f"k={k}: {k}*G = ({result.x}, {result.y}) != P=({P_pt.x}, {P_pt.y})"
    )
