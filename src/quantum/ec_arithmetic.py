import math
from qrisp import (
    QuantumBool,
    QuantumModulus,
    QuantumFloat,
    QuantumArray,
    swap,
    control,
    conjugate,
    invert,
    jrange,
    mcx,
    cx,
    x,
    merge,
    singular_shift,
    custom_control,
    
)


def to_montgomery(x, p):
    n = p.bit_length()
    x *= 2**n % p
    return x


def to_standard(x, p):
    n = p.bit_length()
    x *= pow(2**n, -1, p) % p
    return x


def to_montgomery_qm(x, montgomery_shift):
    
    #N = int(x.modulus)
    R_mod_N = pow(2, montgomery_shift)
    x*=R_mod_N
    
    x.m = montgomery_shift

    
def to_standard_qm(x):
    montgomery_shift = x.m
    x *= pow(2, -montgomery_shift, x.modulus)
    x.m = 0


# Consider moving this function to qrisp source code
def inpl_rsub(r, p):
    x(r)
    r.inpl_adder(r.modulus + 1, r)
    r += p


# Consider moving this function to qrisp source code
# This function is used to compute the modular inverse of a number
def kaliski_quantum(v, p, m):
    n = p.bit_length()
    # Convert to Montgomery
    to_montgomery(v, p) 
    u = QuantumFloat(n)
    u[:] = p
    r = QuantumModulus(2 * p, inpl_adder=v.inpl_adder)
    r[:] = 0
    s = QuantumModulus(2 * p, inpl_adder=v.inpl_adder)
    s[:] = 1

    a = QuantumBool()
    b = QuantumBool()
    add = QuantumBool()
    f = QuantumBool()
    f[:] = True
    for i in jrange(2 * n):
        is_zero = QuantumBool()
        compare_func = lambda x, y: x == y
        injecteted_comp_func = is_zero << compare_func
        with conjugate(injecteted_comp_func)(v, 0):
            mcx([f, is_zero], m[i])
        is_zero.delete()
        cx(m[i], f)
        # STEP 1
        mcx([f, u[0]], a, ctrl_state="10")
        mcx([f, a, v[0]], m[i], ctrl_state="100")
        cx(a, b)
        cx(m[i], b)

        # STEP 2
        l = QuantumBool()
        compare_func = lambda x, y: x > y
        injecteted_comp_func = l << compare_func
        with conjugate(injecteted_comp_func)(u, v):
            mcx([f, l, b], a, ctrl_state="110")
            mcx([f, l, b], m[i], ctrl_state="110")
        l.delete()

        # STEP 3
        with control(a):
            swap(u, v)
            swap(r, s)

        # STEP 4
        mcx([f, b], add, ctrl_state="10")
        with control(add):
            with invert():
                v.inpl_adder(u, v)
            s += r
        # STEP 5
        mcx([f, b], add, ctrl_state="10")
        # uncompute b
        cx(m[i], b)
        cx(a, b)

        # Division by 2
        with control(f):
            with invert():
                singular_shift(v)

        singular_shift(r)

        larger = r > p
        with control(larger):
            r -= p
        cx(r[0], larger[0])
        larger.delete()

        with control(a):
            swap(u, v)
            swap(r, s)
        # uncompute a
        mcx([s[0]], a, ctrl_state="0")

    a.delete()
    add.delete()
    b.delete()

    inpl_rsub(r, p)

    for i in jrange(v.size):
        swap(v[i], r[i])

    # # Uncompute u,s,f
    f.delete()
    x(u[0])
    u.delete()
    r.delete()
    s -= p
    s.delete()
    # Convert back to standard representation
    to_standard(v, p)
    # return v


def qrisp_ell_double(P, curve):
    p = curve.p
    s = ((3 * (P[0] * P[0] % p) + curve.a) % p) * pow((2 * P[1]) % p, -1, p)
    xr = (s * s - 2 * P[0]) % p
    yr = P[1] - s * ((P[0] - xr) % p) % p
    # CHOOSE APPROPRIATE RETURN TYPE
    return [xr, (p - yr) % p]


@custom_control
def qrisp_ell_add_inpl(anc, G, p, ctrl=None):
    # return the result of P + Q
    # Following C3 in the paper
    # Recover p as a Python int since @custom_control may trace it
    p = int(anc[0].modulus)
    mul_func = lambda x, y: x * y

    # Montgomery reduction shift: injection (<<) doesn't propagate .m from
    # the multiplication result, so we set it manually after each injection.
    n_bits = p.bit_length()
    m_red = math.ceil(math.log2((p - 1) ** 2) + 1) - n_bits

    if ctrl is None:
        anc[1] -= G[1]
    else:
        with control(ctrl):
            anc[1] -= G[1]  # step 2 //ctrl_sub_const_modp
    anc[0] -= G[0]  # step 1

    m = QuantumArray(qtype=QuantumBool(), shape=(2 * p.bit_length(),))
    l = QuantumModulus(p, inpl_adder=anc[0].inpl_adder)
    # step 3 & 4 & 6: compute (anc[1] * kaliski_inv(anc[0])) in standard form -> l
    with conjugate(kaliski_quantum)(anc[0], p, m):
        temp = QuantumModulus(p)
        injected_mul = temp << mul_func
        with conjugate(injected_mul)(anc[1], anc[0]):
            temp.m = -m_red
            with conjugate(to_standard_qm)(temp):
                cx(temp, l)
        temp.delete()
    m.delete()

    # step 5: anc[1] -= l * anc[0] (in standard form)
    temp = QuantumModulus(p)
    injected_mul = temp << mul_func
    with conjugate(injected_mul)(l, anc[0]):
        temp.m = -m_red
        with conjugate(to_standard_qm)(temp):
            anc[1] -= temp
    temp.delete()

    if ctrl is None:
        anc[0] += 3 * G[0]
    else:
        with control(ctrl):
            anc[0] += 3 * G[0]  # step 9 //ctrl_add_const_modp

    # step 7 & 8: anc[0] -= l * l (in standard form)
    temp = QuantumModulus(p)
    injected_mul = temp << mul_func
    with conjugate(injected_mul)(l, l):
        temp.m = -m_red
        with conjugate(to_standard_qm)(temp):
            if ctrl is None:
                anc[0] -= temp  # step 8
            else:
                with control(ctrl):
                    anc[0] -= temp  # step 8 //ctrl_sub_modp
    temp.delete()  # step 10

    # step 11: anc[1] += l * anc[0] (in standard form)
    temp = QuantumModulus(p)
    injected_mul = temp << mul_func
    with conjugate(injected_mul)(l, anc[0]):
        temp.m = -m_red
        with conjugate(to_standard_qm)(temp):
            anc[1] += temp
    temp.delete()

    # step 12-14: uncompute l via second kaliski inversion
    m = QuantumArray(qtype=QuantumBool(), shape=(2 * p.bit_length(),))
    with conjugate(kaliski_quantum)(anc[0], p, m):
        temp = QuantumModulus(p)
        injected_mul = temp << mul_func
        with conjugate(injected_mul)(anc[1], anc[0]):
            temp.m = -m_red
            with conjugate(to_standard_qm)(temp):
                cx(temp, l)
        temp.delete()
    m.delete()

    l.delete()
    if ctrl is None:
        anc[1] -= G[1]  # step 16
    else:
        with control(ctrl):
            anc[1] -= G[1]  # step 16 //ctrl_sub_const_modp

    anc[0] -= G[0]  # step 17

    if ctrl is None:
        inpl_rsub(anc[0], p)  # step 15
    else:
        with control(ctrl):
            inpl_rsub(anc[0], p)  # step 15 //ctrl_neg_modp


def qrisp_ell_mult_add(power, res, k, curve):
    # Elliptic curve multiplication Q + kP
    n = k.size
    p = curve.p
    merge([res, k])

    for i in range(n):
        with control(k[i]):
            res = qrisp_ell_add_inpl(res, power, p)
        # with invert():
        # cyclic_shift(k)
        power = qrisp_ell_double(power, curve)
    return res
