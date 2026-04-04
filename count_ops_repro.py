"""
Reproducible example: @count_ops profiling is extremely slow for deeply
nested circuits used in EC arithmetic.

`make_jaspr` (tracing only, no profiling) is fast because it only records
the circuit structure as a Jaspr IR without evaluating gate counts.

Usage:
    python count_ops_repro.py               # run all tests
    python count_ops_repro.py kaliski       # run only kaliski test
    python count_ops_repro.py add           # run only q_ec_add_inpl test
    python count_ops_repro.py ctrl_add      # run only controlled addition test
"""

import sys
import time
from qrisp import (
    QuantumModulus, QuantumBool, QuantumArray, QuantumFloat,
    measure, gidney_adder, count_ops, h, control
)
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
from qrisp.jasp import make_jaspr

sys.path.insert(0, ".")
import src.quantum.ec_arithmetic as qECarithm

# --- Setup: 4-bit prime p=13, curve y²=x³+7 (mod 13) ---
P_INT = 13
N_BITS = 4
P_BI = BigInteger.create_static(P_INT, 1)

# A known point on y²=x³+7 (mod 13): (11, 5)
G = (11, 5)
G0_BI = BigInteger.create_static(G[0], 1)
G1_BI = BigInteger.create_static(G[1], 1)


def run_test(name, make_jaspr_fn, count_ops_fn, count_ops_timeout=300):
    """Run a single test: first make_jaspr (fast), then count_ops (slow)."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # --- Phase 1: make_jaspr (tracing only) ---
    print(f"  make_jaspr ...", end=" ", flush=True)
    t0 = time.time()
    jaspr = make_jaspr(make_jaspr_fn)()
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s — {len(jaspr.eqns)} top-level eqns")

    # --- Phase 2: count_ops (profiling) ---
    print(f"  count_ops  ...", end=" ", flush=True)
    t0 = time.time()
    try:
        ops = count_ops_fn()
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")
        for k, v in sorted(ops.items()):
            print(f"    {k}: {v}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED in {elapsed:.1f}s: {type(e).__name__}: {e}")

    return elapsed


# ============================================================
# Test 1: kaliski_mod_inv (the inner bottleneck)
# ============================================================
def kaliski_circuit():
    v = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    v[:] = G0_BI
    m = QuantumArray(qtype=QuantumBool(), shape=(2 * N_BITS,))
    qECarithm.kaliski_mod_inv(v, P_INT, m)
    measure(v)


@count_ops(meas_behavior="0")
def kaliski_count():
    v = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    v[:] = G0_BI
    m = QuantumArray(qtype=QuantumBool(), shape=(2 * N_BITS,))
    qECarithm.kaliski_mod_inv(v, P_INT, m)
    measure(v)


# ============================================================
# Test 2: q_ec_add_inpl (uncontrolled) — calls kaliski twice
# ============================================================
def add_circuit():
    res_0 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_0[:] = G0_BI
    res_1 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_1[:] = G1_BI
    qECarithm.q_ec_add_inpl([res_0, res_1], G, P_BI, ctrl=None)
    measure(res_0)
    measure(res_1)


@count_ops(meas_behavior="0")
def add_count():
    res_0 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_0[:] = G0_BI
    res_1 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_1[:] = G1_BI
    qECarithm.q_ec_add_inpl([res_0, res_1], G, P_BI, ctrl=None)
    measure(res_0)
    measure(res_1)


# ============================================================
# Test 3: q_ec_add_inpl (controlled) — adds control overhead
# ============================================================
def ctrl_add_circuit():
    res_0 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_0[:] = G0_BI
    res_1 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_1[:] = G1_BI
    c = QuantumBool()
    h(c)
    qECarithm.q_ec_add_inpl([res_0, res_1], G, P_BI, ctrl=c)
    measure(res_0)
    measure(res_1)


@count_ops(meas_behavior="0")
def ctrl_add_count():
    res_0 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_0[:] = G0_BI
    res_1 = QuantumModulus(P_BI, inpl_adder=gidney_adder)
    res_1[:] = G1_BI
    c = QuantumBool()
    h(c)
    qECarithm.q_ec_add_inpl([res_0, res_1], G, P_BI, ctrl=c)
    measure(res_0)
    measure(res_1)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    tests = {
        "kaliski":  ("kaliski_mod_inv (4-bit BigInteger)", kaliski_circuit, kaliski_count),
        "add":      ("q_ec_add_inpl uncontrolled (4-bit BigInteger)", add_circuit, add_count),
        "ctrl_add": ("q_ec_add_inpl controlled (4-bit BigInteger)", ctrl_add_circuit, ctrl_add_count),
    }

    # Parse CLI: run specific test or all
    requested = sys.argv[1] if len(sys.argv) > 1 else None
    if requested and requested not in tests:
        print(f"Unknown test '{requested}'. Available: {', '.join(tests.keys())}")
        sys.exit(1)

    results = {}
    for key, (name, jaspr_fn, ops_fn) in tests.items():
        if requested and key != requested:
            continue
        elapsed = run_test(name, jaspr_fn, ops_fn)
        results[key] = elapsed

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for key, elapsed in results.items():
        print(f"  {key}: count_ops took {elapsed:.1f}s")
