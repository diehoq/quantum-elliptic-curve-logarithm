"""Inversion-free x-only (projective X:Z) elliptic-curve arithmetic in Qrisp.

This is an alternative to the affine engine in `ec_arithmetic.py`.  The affine
`q_ec_add_inpl` runs `kaliski_mod_inv` twice per point addition; a Montgomery
ladder built from the primitives below uses NO modular inversion in the loop
(only field mul/sqr/add), deferring a single inversion to the final X/Z
deaffinify after measurement.  See `src/classical/xonly_reference.py` for the
classical ground truth and the inversion-count comparison.

Curve: short Weierstrass y^2 = x^3 + a x + b, a = 0 path (secp256k1 / QDay family).

Qrisp version note
------------------
Validated on qrisp 0.9.5 (PyPI).  The montgomery-shift behaviour of
QuantumModulus differs between releases; three rules (discovered empirically and
documented inline) make multiply/add/sub compose correctly under
`boolean_simulation` on 0.9.5:

  1. Plain `A*B` decodes correctly and chains; its montgomery shift accumulates
     as m(A*B) = m(A)+m(B)-m_red.  A classical const-mult `C*int` RESETS shift
     to 0 (it is the normalizer).
  2. `+`/`-` require equal shift: operands at the SAME depth are combined
     directly; when depths differ, normalize exactly ONE side via `_std` (a `*1`).
     Do NOT normalize both sides of a subtraction (`_std(A)-_std(B)` is wrong).
  3. Never multiply a register by itself (`C*C`): squarings use two INDEPENDENT
     products, e.g. `(X1*X2)*(X1*X2)`.

Under 0.8.2 (this repo's pinned branch) QuantumModulus has a montgomery
boolean_simulation uncomputation bug (the repo's own boolean_sim test exhibits
it), so these primitives should be run against qrisp >= 0.9.5.
"""
from qrisp import QuantumModulus, control, swap


def _std(C):
    """Normalize montgomery shift to 0 via a classical *1 (preserves value)."""
    return C * 1


def q_xDBL(X1, Z1, a, b, p):
    """x-only projective doubling (a=0). Returns (X2, Z2). Inversion-free."""
    c8b = (8 * b) % p
    bm = b % p
    X1sq = X1 * X1
    Z1sq = Z1 * Z1
    Z1cu = Z1sq * Z1
    X1cu = X1sq * X1
    X1q = X1sq * X1sq          # X1^4   (shift -3 m_red, same as X1Z1cu)
    X1Z1cu = X1 * Z1cu         # X1 Z1^3
    X2 = _std(X1q) - X1Z1cu * c8b      # one-sided _std vs const-product
    inner = _std(X1cu) + Z1cu * bm     # one-sided _std vs const-product
    Z2 = (Z1 * inner) * (4 % p)
    return X2, Z2


def q_xADD(X1, Z1, X2, Z2, xD, a, b, p):
    """x-only differential addition (a=0, difference Zd=1, Xd=xD). Returns (Xp, Zp)."""
    c4b = (4 * b) % p
    sq = (X1 * X2) * (X1 * X2)          # (X1 X2)^2 via two independent products
    Z1Z2 = Z1 * Z2
    X1Z2 = X1 * Z2
    X2Z1 = X2 * Z1
    sumXZ = X1Z2 + X2Z1                 # same shift -> direct
    term = (Z1Z2 * sumXZ) * c4b
    Xp = _std(sq) - term
    diffA = (X1 * Z2) - (X2 * Z1)       # two independent differences (no self-mul)
    diffB = (X1 * Z2) - (X2 * Z1)
    Zp = (diffA * diffB) * (xD % p)
    return Xp, Zp


def q_ladder_x(k, xG, twoGx, a, b, p):
    """Classical scalar k: full Montgomery ladder, returns projective (X0:Z0)=x([k]G).
    Inversion-free loop. `twoGx` = affine x(2G)."""
    n = k.bit_length()
    X0 = QuantumModulus(p); X0[:] = xG
    Z0 = QuantumModulus(p); Z0[:] = 1
    X1 = QuantumModulus(p); X1[:] = twoGx
    Z1 = QuantumModulus(p); Z1[:] = 1
    for i in range(n - 2, -1, -1):
        if (k >> i) & 1:
            X0, Z0 = q_xADD(X0, Z0, X1, Z1, xG, a, b, p)
            X1, Z1 = q_xDBL(X1, Z1, a, b, p)
        else:
            X1, Z1 = q_xADD(X0, Z0, X1, Z1, xG, a, b, p)
            X0, Z0 = q_xDBL(X0, Z0, a, b, p)
    return X0, Z0


def _cswap(ctrl, A, B):
    with control(ctrl):
        swap(A, B)


def q_ladder_quantum(kbits_low, xG, twoGx, a, b, p):
    """Quantum scalar (real Shor form): the scalar bits drive CONTROLLED-SWAPs of
    the ladder points instead of classical branching.  With the bits in
    superposition the same circuit computes x([k]G) for all k at once.

    Assumes the MSB is 1 (true for Bitcoin-puzzle keys k in [2^(n-1), 2^n)), so
    R0=G, R1=2G are the post-MSB state; `kbits_low` are the QuantumBools for the
    remaining bits, ordered MSB-1 .. 0.  Returns projective (X0:Z0)=x([k]G)."""
    X0 = QuantumModulus(p); X0[:] = xG
    Z0 = QuantumModulus(p); Z0[:] = 1
    X1 = QuantumModulus(p); X1[:] = twoGx
    Z1 = QuantumModulus(p); Z1[:] = 1
    for kbit in kbits_low:
        _cswap(kbit, X0, X1); _cswap(kbit, Z0, Z1)
        nX1, nZ1 = q_xADD(X0, Z0, X1, Z1, xG, a, b, p)   # R1 <- R0 + R1
        nX0, nZ0 = q_xDBL(X0, Z0, a, b, p)               # R0 <- 2 R0
        _cswap(kbit, nX0, nX1); _cswap(kbit, nZ0, nZ1)
        X0, Z0, X1, Z1 = nX0, nZ0, nX1, nZ1
    return X0, Z0
