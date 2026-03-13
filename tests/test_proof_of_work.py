"""
Proof-of-work tests for the QDay EC discrete logarithm challenge.

All curves: y² = x³ + 7  mod p   (a=0, b=7)
Test data loaded from curves_and_keys.json.

Tier 1 — Classical verification: confirm k·G = P for every entry.
Tier 2 — Resource estimation: build the Shor circuit and compile it to
         extract qubit count, T-count, CNOT count, and T-depth.
         No simulation is needed — compilation alone is the proof of work.
Tier 3 — Functional simulation: run Shor end-to-end on the 4-bit curve.
"""
import json
import os

import qrisp
import pytest
import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm


# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_JSON = os.path.join(_HERE, os.pardir, "curves_and_keys.json")

with open(_JSON) as _f:
    CURVES_AND_KEYS = json.load(_f)

A, B = 0, 7  # all curves share y² = x³ + 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_shor_circuit(entry):
    """Build the full Shor ECDLP circuit for a given curve entry.
    Returns (anc, x1, x2) — the circuit lives on anc[0].qs."""
    p = entry["prime"]
    curve = clECarithm.EllCurve(A, B, p)
    G = list(entry["generator_point"])
    P = list(entry["public_key"])

    n_bits = p.bit_length()
    x1 = qrisp.QuantumModulus(2**n_bits, inpl_adder=qrisp.gidney_adder)
    x2 = qrisp.QuantumModulus(2**n_bits, inpl_adder=qrisp.gidney_adder)

    mod_p = qrisp.QuantumModulus(p, inpl_adder=qrisp.gidney_adder)
    anc = qrisp.QuantumArray(qtype=mod_p, shape=(2,))
    anc[:] = G  # initialise to generator (a known curve point)

    qrisp.h(x1)
    qrisp.h(x2)

    anc = qECarithm.q_ec_mult_add(G, anc, x1, curve)
    anc = qECarithm.q_ec_mult_add(P, anc, x2, curve)

    qrisp.QFT(x1, inv=True)
    qrisp.QFT(x2, inv=True)

    return anc, x1, x2


def _t_depth_indicator(op):
    return 1 if op.name in ("t", "t_dg") else 0


def _compile_and_measure(anc):
    """Compile the circuit and return resource metrics (no simulation)."""
    compiled = anc[0].qs.compile(
        compile_mcm=True,
        gate_speed=_t_depth_indicator,
        workspace=anc[0].size,
    )
    qubit_count = compiled.num_qubits()
    gate_counts = compiled.transpile().count_ops()
    cnot_count = gate_counts.get("cx", 0) + 0.5 * gate_counts.get("c_if_cz", 0)
    t_count = gate_counts.get("t", 0) + gate_counts.get("t_dg", 0)
    t_depth = compiled.depth(depth_indicator=_t_depth_indicator)
    return {
        "qubit_count": qubit_count,
        "t_count": t_count,
        "cnot_count": cnot_count,
        "t_depth": t_depth,
    }


# ---------------------------------------------------------------------------
# Tier 1 — Classical verification (all 17 curves)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "entry", CURVES_AND_KEYS, ids=lambda e: f"{e['bit_length']}bit"
)
def test_classical_verify(entry):
    """Verify G and P are on the curve and k·G = P."""
    p = entry["prime"]
    curve = clECarithm.EllCurve(A, B, p)
    xG, yG = entry["generator_point"]
    xP, yP = entry["public_key"]
    k = entry["private_key"]

    # Points on curve
    assert (yG * yG - xG**3 - A * xG - B) % p == 0, "G not on curve"
    assert (yP * yP - xP**3 - A * xP - B) % p == 0, "P not on curve"

    # k·G == P
    G_pt = clECarithm.EllPoint(xG, yG)
    result = G_pt
    for _ in range(k - 1):
        result = clECarithm.ell_add(result, G_pt, curve)
    assert (result.x, result.y) == (xP, yP), (
        f"k·G = ({result.x},{result.y}) != P = ({xP},{yP})"
    )


# ---------------------------------------------------------------------------
# Tier 2 — Resource estimation (circuit build + compile, no simulation)
#
# Extend _RESOURCE_BIT_LENGTHS as hardware/time budget allows.
# ---------------------------------------------------------------------------
_RESOURCE_BIT_LENGTHS = [4, 6, 7, 8]

_resource_entries = [
    e for e in CURVES_AND_KEYS if e["bit_length"] in _RESOURCE_BIT_LENGTHS
]


@pytest.mark.parametrize(
    "entry", _resource_entries, ids=lambda e: f"{e['bit_length']}bit"
)
def test_resource_estimation(entry):
    """Build and compile the full Shor circuit; report resource metrics."""
    anc, x1, x2 = _build_shor_circuit(entry)
    metrics = _compile_and_measure(anc)

    assert metrics["qubit_count"] > 0
    assert metrics["t_count"] >= 0

    print(
        f"\n  {entry['bit_length']}-bit (p={entry['prime']}): "
        f"qubits={metrics['qubit_count']}, "
        f"T-count={metrics['t_count']}, "
        f"CNOT={metrics['cnot_count']}, "
        f"T-depth={metrics['t_depth']}"
    )


# ---------------------------------------------------------------------------
# Tier 3 — Full simulation (4-bit curve only)
# ---------------------------------------------------------------------------
def test_shor_simulation_4bit():
    """Simulate Shor's algorithm on the 4-bit curve and recover the key."""
    entry = next(e for e in CURVES_AND_KEYS if e["bit_length"] == 4)
    p = entry["prime"]
    curve = clECarithm.EllCurve(A, B, p)
    G = list(entry["generator_point"])
    P = list(entry["public_key"])
    expected_k = entry["private_key"]
    order = entry["subgroup_order"]

    n_bits = p.bit_length()
    x1 = qrisp.QuantumModulus(2**n_bits)
    x2 = qrisp.QuantumModulus(2**n_bits)

    mod_p = qrisp.QuantumModulus(p)
    anc = qrisp.QuantumArray(qtype=mod_p, shape=(2,))
    anc[:] = G

    qrisp.h(x1)
    qrisp.h(x2)

    anc = qECarithm.q_ec_mult_add(G, anc, x1, curve)
    anc = qECarithm.q_ec_mult_add(P, anc, x2, curve)

    qrisp.QFT(x1, inv=True)
    qrisp.QFT(x2, inv=True)

    results = qrisp.multi_measurement([x1, x2])

    # Extract discrete log from measurement outcomes:
    # For each (a, b) with b != 0 mod order: k = -a·b⁻¹ mod order
    G_pt = clECarithm.EllPoint(*G)
    P_pt = clECarithm.EllPoint(*P)
    recovered = set()
    for (a_val, b_val), _ in results.items():
        if b_val % order == 0:
            continue
        try:
            k_candidate = (-a_val * pow(b_val, -1, order)) % order
        except (ValueError, ZeroDivisionError):
            continue
        # Verify classically
        check = G_pt
        for _ in range(k_candidate - 1):
            check = clECarithm.ell_add(check, G_pt, curve)
        if (check.x, check.y) == (P_pt.x, P_pt.y):
            recovered.add(k_candidate)

    assert expected_k in recovered, (
        f"Failed to recover k={expected_k}. "
        f"Candidates: {recovered}, outcomes: {results}"
    )
