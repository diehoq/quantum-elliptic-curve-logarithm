import qrisp


def to_montgomery(x, p):
    n = p.bit_length()
    x *= 2**n % p
    return x


def to_standard(x, p):
    n = p.bit_length()
    x *= pow(2**n, -1, p) % p
    return x


def to_montgomery_qm(x, montgomery_shift):
    x *= pow(2, montgomery_shift, x.modulus)
    x.m = montgomery_shift


def to_standard_qm(x):
    montgomery_shift = x.m
    x *= pow(2, -montgomery_shift, x.modulus)
    x.m = 0


# Consider moving this function to qrisp source code
def inpl_rsub(r, p):
    qrisp.x(r)
    r.inpl_adder(r.modulus + 1, r)
    r += p


# Consider moving this function to qrisp source code
# This function is used to compute the modular inverse of a number
def kaliski_quantum(v, p, m):
    n = p.bit_length()
    # Convert to Montgomery
    to_montgomery(v, p)
    u = qrisp.QuantumFloat(n)
    u[:] = p
    r = qrisp.QuantumModulus(2 * p, inpl_adder=v.inpl_adder)
    r[:] = 0
    s = qrisp.QuantumModulus(2 * p, inpl_adder=v.inpl_adder)
    s[:] = 1

    v.__class__ = qrisp.QuantumFloat

    a = qrisp.QuantumBool()
    b = qrisp.QuantumBool()
    add = qrisp.QuantumBool()
    f = qrisp.QuantumBool()
    f[:] = True
    for i in range(2 * n):
        is_zero = v == 0
        qrisp.mcx([f, is_zero], m[i])
        is_zero.uncompute()
        qrisp.cx(m[i], f)
        # STEP 1
        qrisp.mcx([f, u[0]], a, ctrl_state="10")
        qrisp.mcx([f, a, v[0]], m[i], ctrl_state="100")
        qrisp.cx(a, b)
        qrisp.cx(m[i], b)

        # STEP 2
        l = u > v
        qrisp.mcx([f, l, b], a, ctrl_state="110")
        qrisp.mcx([f, l, b], m[i], ctrl_state="110")
        l.uncompute()

        # STEP 3
        with qrisp.control(a):
            qrisp.swap(u, v)
            qrisp.swap(r, s)

        # STEP 4
        qrisp.mcx([f, b], add, ctrl_state="10")
        with qrisp.control(add):
            v -= u
            s += r
        # STEP 5
        qrisp.mcx([f, b], add, ctrl_state="10")
        # uncompute b
        qrisp.cx(m[i], b)
        qrisp.cx(a, b)

        # Division by 2
        with qrisp.control(f):
            with qrisp.invert():
                qrisp.cyclic_shift(v)

        qrisp.cyclic_shift(r)
        larger = r > p
        with larger:
            r -= p
        qrisp.cx(r[0], larger)
        larger.delete()

        with qrisp.control(a):
            qrisp.swap(u, v)
            qrisp.swap(r, s)
        # uncompute a
        qrisp.mcx([s[0]], a, ctrl_state="0")

    a.delete()
    add.delete()
    b.delete()

    inpl_rsub(r, p)

    v.__class__ = qrisp.QuantumModulus
    for i in range(v.size):
        qrisp.swap(v[i], r[i])

    # Uncompute u,s,f
    f.delete()
    qrisp.x(u[0])
    u.delete()
    r.delete()
    s -= p
    s.delete()
    # Convert back to standard representation
    to_standard(v, p)
    return v


def qrisp_ell_double(P, curve):
    p = curve.p
    s = ((3 * (P[0] * P[0] % p) + curve.a) % p) * pow((2 * P[1]) % p, -1, p)
    xr = (s * s - 2 * P[0]) % p
    yr = P[1] - s * ((P[0] - xr) % p) % p
    # CHOOSE APPROPRIATE RETURN TYPE
    return [xr, (p - yr) % p]


@qrisp.custom_control
def qrisp_ell_add_inpl(anc, G, p, ctrl=None):
    # return the result of P + Q
    # Following C3 in the paper
    if ctrl is None:
        anc[1] -= G[1]
    else:
        with qrisp.control(ctrl):
            anc[1] -= G[1]  # step 2 //ctrl_sub_const_modp
    anc[0] -= G[0]  # step 1

    m = qrisp.QuantumArray(qtype=qrisp.QuantumBool(), shape=(2 * p.bit_length(),))
    l = qrisp.QuantumModulus(p, inpl_adder=anc[0].inpl_adder)
    with qrisp.conjugate(kaliski_quantum)(anc[0], p, m) as inv:
        temp = anc[1] * inv
        to_standard_qm(temp)
        l[:] = temp  # step 3 & 4 & 6
        temp.uncompute()
    for a in m:
        a.delete()
    # step 5
    temp = l * anc[0]
    to_standard_qm(temp)
    anc[1] -= temp
    temp.uncompute()
    # anc[1].delete()

    if ctrl is None:
        anc[0] += 3 * G[0]
    else:
        with qrisp.control(ctrl):
            anc[0] += 3 * G[0]  # step 9 //ctrl_add_const_modp

    temp = l * l  # step 7
    to_standard_qm(temp)
    if ctrl is None:
        anc[0] -= temp  # step 8
    else:
        with qrisp.control(ctrl):
            anc[0] -= temp  # step 8 //ctrl_sub_modp
    temp.uncompute()  # step 10

    # step 11
    temp = l * anc[0]
    to_standard_qm(temp)
    anc[1] += temp
    temp.uncompute()

    m = qrisp.QuantumArray(qtype=qrisp.QuantumBool(), shape=(2 * p.bit_length(),))

    with qrisp.conjugate(kaliski_quantum)(anc[0], p, m) as inv:
        temp = anc[1] * inv
        to_standard_qm(temp)
        qrisp.cx(temp, l)
        temp.uncompute()

    for a in m:
        a.delete()

    l.delete()
    if ctrl is None:
        anc[1] -= G[1]  # step 16
    else:
        with qrisp.control(ctrl):
            anc[1] -= G[1]  # step 16 //ctrl_sub_const_modp

    anc[0] -= G[0]  # step 17

    if ctrl is None:
        inpl_rsub(anc[0], p)  # step 15
    else:
        with qrisp.control(ctrl):
            inpl_rsub(anc[0], p)  # step 15 //ctrl_neg_modp

    return anc


def qrisp_ell_mult_add(power, res, k, curve):
    # Elliptic curve multiplication Q + kP
    n = k.size
    p = curve.p
    qrisp.merge([res, k])
    with qrisp.IterationEnvironment(res.qs, n, precompile=True):
        with qrisp.control(k[0]):
            res = qrisp_ell_add_inpl(res, power, p)
        with qrisp.invert():
            qrisp.cyclic_shift(k)
        power = qrisp_ell_double(power, curve)
    return res
