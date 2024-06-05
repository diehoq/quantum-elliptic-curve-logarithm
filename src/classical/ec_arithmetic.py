# Elliptic curve class
class EllCurve:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p


# Elliotic curve point clqss
class EllPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Neutral element
EllZero = EllPoint(0, 0)


# Function to compute y on EC given x modulo p
def compute_y(x, a, b, p):
    for y in range(p):
        if (y**2 % p) == ((x**3 + a * x + b) % p):
            return y
    return None


# return the result of P + Q
def ell_add_generic(P, Q, curve):
    p = curve.p
    s = (((P.y - Q.y) % p) * pow((P.x - Q.x) % p, -1, p)) % p
    xr = p + (s * s - P.x - Q.x) % p
    yr = (P.y - s * ((P.x - xr) % p)) % p
    return EllPoint((p + xr) % p, (p - yr) % p)


# return the result of 2*P
def ell_double(P, curve):
    p = curve.p
    s = ((3 * (P.x * P.x % p) + curve.a) % p) * pow((2 * P.y) % p, -1, p)
    xr = (s * s - 2 * P.x) % p
    yr = P.y - s * ((P.x - xr) % p) % p
    return EllPoint((p + xr) % p, (p - yr) % p)


# return the result of Q + P considering the special cases
def ell_add(P, Q, curve):
    p = curve.p
    if P.x == EllZero.x and P.y == EllZero.y:
        return Q
    if Q.x == EllZero.x and Q.y == EllZero.y:
        return P
    if P.x == Q.x:
        if P.y == (p - Q.y) % p:
            return EllZero
        if P.y == Q.y and Q.y != 0:
            return ell_double(P, curve)
    return ell_add_generic(P, Q, curve)


# return the result of Q + k*P
def ell_mult_add(P, Q, k, curve):
    n = k.bit_length()
    res = Q
    power = P
    for _ in range(n):
        if k % 2:
            res = ell_add(res, power, curve)
        k >>= 1
        power = ell_double(power, curve)
    return res


def compute_logarithm(G, P, curve):
    temp = G
    p = curve.p
    # Find the right range instead of 2*p
    for i in range(1, 2 * p):
        if temp.x == P.x and temp.y == P.y:
            return i
        temp = ell_add(temp, G, curve)

    return None
