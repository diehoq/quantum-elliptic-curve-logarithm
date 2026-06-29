"""Classical x-only (projective Kummer-line) reference for short Weierstrass
curves y^2 = x^3 + a*x + b  (the QDay / secp256k1 family uses a=0, b=7).

Motivation
----------
The quantum engine in `src/quantum/ec_arithmetic.py` performs scalar
multiplication by affine double-and-add: every controlled point addition
(`q_ec_add_inpl`) computes a slope and therefore runs `kaliski_mod_inv` TWICE
(compute lambda + uncompute lambda).  Modular inversion is by far the most
expensive reversible primitive (Kaliski = 2n loop iterations), so an n-bit scalar
costs on the order of ~4n inversions on the critical path.

The Montgomery ladder in projective (X:Z) coordinates removes every intermediate
inversion: each step is built only from field multiplications/squarings/additions
(xDBL / xADD), and a single inversion at the very end deaffinifies x = X/Z.  This
module is the inversion-free reference; `tests/test_xonly_reference.py` validates
it against this repo's own affine group law on every curve in
`curves_and_keys.json` and quantifies the inversion saving.

The quantum counterpart lives in `src/quantum/ec_arithmetic_xonly.py`.
"""


class FieldCounter:
    """Prime field with operation counting (mul/sqr/add/inv) so we can quantify
    the resource profile of each scalar-multiplication strategy."""

    def __init__(self, p):
        self.p = p
        self.mul = self.sqr = self.add = self.inv = 0

    def m(self, a, b):
        self.mul += 1
        return (a * b) % self.p

    def s(self, a):
        self.sqr += 1
        return (a * a) % self.p

    def add_(self, a, b):
        self.add += 1
        return (a + b) % self.p

    def sub(self, a, b):
        self.add += 1
        return (a - b) % self.p

    def cmul(self, k, a):
        self.add += 1            # multiply by a small classical constant (cheap)
        return (k * a) % self.p

    def inverse(self, a):
        self.inv += 1
        return pow(a, -1, self.p)


# ----------------------------------------------------------------------------
# Projective x-only formulas (Izu-Takagi / Brier-Joye), short Weierstrass.
# Difference point fixed = base point (xD : 1).
# ----------------------------------------------------------------------------
def xDBL(X1, Z1, a, b, F):
    """(X1:Z1) -> x([2]P).  No inversion."""
    X1sq = F.s(X1)
    Z1sq = F.s(Z1)
    aZ1sq = F.m(a, Z1sq) if a else 0
    t = F.sub(X1sq, aZ1sq)
    X2 = F.s(t)                                   # (X1^2 - a Z1^2)^2
    Z1cu = F.m(Z1sq, Z1)
    bX1Z1cu = F.m(b, F.m(X1, Z1cu))
    X2 = F.sub(X2, F.cmul(8, bX1Z1cu))            # - 8 b X1 Z1^3
    X1cu = F.m(X1sq, X1)
    inner = X1cu
    if a:
        inner = F.add_(inner, F.m(a, F.m(X1, Z1sq)))
    inner = F.add_(inner, F.m(b, Z1cu))           # X1^3 + a X1 Z1^2 + b Z1^3
    Z2 = F.cmul(4, F.m(Z1, inner))
    return X2, Z2


def xADD(X1, Z1, X2, Z2, xD, a, b, F):
    """Differential add (X1:Z1)+(X2:Z2) with known x(P-Q)=xD (Zd=1).  No inversion."""
    X1X2 = F.m(X1, X2)
    Z1Z2 = F.m(Z1, Z2)
    X1Z2 = F.m(X1, Z2)
    X2Z1 = F.m(X2, Z1)
    t1 = F.sub(X1X2, F.m(a, Z1Z2)) if a else X1X2
    t1 = F.s(t1)
    t2 = F.m(F.cmul(4, F.m(b, Z1Z2)), F.add_(X1Z2, X2Z1))
    Xp = F.sub(t1, t2)
    Zp = F.m(xD, F.s(F.sub(X1Z2, X2Z1)))
    return Xp, Zp


def ladder_x(k, xP, a, b, F):
    """Return affine x([k]P) via the x-only ladder.  ONE inversion at the end."""
    n = k.bit_length()
    if n == 0:
        return None
    X0, Z0 = xP, 1
    X1, Z1 = xDBL(xP, 1, a, b, F)
    for i in range(n - 2, -1, -1):
        if (k >> i) & 1:
            X0, Z0 = xADD(X0, Z0, X1, Z1, xP, a, b, F)
            X1, Z1 = xDBL(X1, Z1, a, b, F)
        else:
            X1, Z1 = xADD(X0, Z0, X1, Z1, xP, a, b, F)
            X0, Z0 = xDBL(X0, Z0, a, b, F)
    if Z0 == 0:
        return None
    return F.m(X0, F.inverse(Z0))


def affine_doubleadd_inversions(k):
    """Number of modular inversions an affine double-and-add scalar mult uses
    (one per non-trivial add/double) -- the cost driver of the current quantum
    q_ec_add_inpl, which turns each into TWO Kaliski inversions."""
    invs = 0
    has_R = False
    Q_is_zero = False
    bits = k.bit_length()
    for i in range(bits):
        if (k >> i) & 1:
            if has_R:
                invs += 1          # affine addition
            has_R = True
        if i < bits - 1:
            invs += 1              # doubling
    return invs
