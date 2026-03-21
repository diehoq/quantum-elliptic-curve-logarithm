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
    qache,
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

    # N = int(x.modulus)
    R_mod_N = pow(2, montgomery_shift)
    x *= R_mod_N

    x.m = montgomery_shift


def to_standard_qm(x):
    montgomery_shift = x.m
    x *= pow(2, -montgomery_shift, x.modulus)
    x.m = 0


# Consider moving this function to qrisp source code
def inpl_rsub(r, p):
    # Compute r := (p - r) mod p.
    # The bit-trick (NOT + add modulus+1 + modular-add p) creates intermediate
    # value == modulus when r=0, which breaks mod_adder (its ancilla can't
    # uncompute because a=0 makes forward/reverse additions all no-ops).
    # This only matters when p % r.modulus == 0 (i.e. modulus == p), because
    # then __iadd__ pre-reduces p to 0 and mod_adder receives a=0.
    # For modulus == 2p, the add is non-degenerate and mod_adder works fine.
    if p % int(r.modulus) == 0:
        # r += p reduces to mod_adder(0,...) which faults on value=modulus.
        # Fix: skip when r=0 (it's a no-op since (p-0) mod p = 0).
        is_zero = QuantumBool()
        eq = is_zero << (lambda a, b: a == b)
        with conjugate(eq)(r, 0):
            with control(is_zero, ctrl_state="0"):
                x(r)
                r.inpl_adder(r.modulus + 1, r)
                r += p
        is_zero.delete()
    else:
        x(r)
        r.inpl_adder(r.modulus + 1, r)
        r += p


# Consider moving this function to qrisp source code
# This function is used to compute the modular inverse of a number
@qache(static_argnums=[1])
def kaliski_mod_inv(v, p, m):
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

    # Fix for v=0 input (phase 1): the loop is a no-op, leaving
    # u=p, v=0, r=0, s=1.  Detect via u==p (u=1 for all v!=0)
    # and neutralise r so that inpl_rsub(p,p)=0, keeping v=0 after swap.
    v_was_zero = QuantumBool()
    eq_u_p = v_was_zero << (lambda a, b: a == b)
    with conjugate(eq_u_p)(u, p):
        with control(v_was_zero):
            r += p
    v_was_zero.delete()

    inpl_rsub(r, p)

    for i in jrange(v.size):
        swap(v[i], r[i])

    # Fix for v=0 input (phase 2): after phase 1, v=0 uniquely identifies
    # the original-v=0 case (v!=0 always lands in [1,p-1]).
    # Cleanup expects u=1, s=p but we still have u=p, s=1.  XOR-correct.
    v_was_zero = QuantumBool()
    eq_v_0 = v_was_zero << (lambda a, b: a == b)
    with conjugate(eq_v_0)(v, 0):
        with control(v_was_zero):
            correction = p ^ 1
            for i in range(n):
                if (correction >> i) & 1:
                    x(u[i])
                    x(s[i])
    v_was_zero.delete()

    # Uncompute u,s,f
    f.delete()
    x(u[0])
    u.delete()
    r.delete()
    s -= p
    s.delete()
    # Convert back to standard representation
    to_standard(v, p)
    # return v


def q_ec_double(P, curve):
    p = curve.p
    s = ((3 * (P[0] * P[0] % p) + curve.a) % p) * pow((2 * P[1]) % p, -1, p)
    xr = (s * s - 2 * P[0]) % p
    yr = P[1] - s * ((P[0] - xr) % p) % p
    # CHOOSE APPROPRIATE RETURN TYPE
    return [xr, (p - yr) % p]


@custom_control
def q_ec_add_inpl(anc, G, p, ctrl=None):
    # return the result of P + Q
    # Following C3 in the paper
    # Recover p as a Python int since @custom_control may trace it
    p = int(anc[0].modulus)
    mul_func = lambda x, y: x * y

    # Montgomery reduction shift: injection (<<) doesn't propagate .m from
    # the multiplication result, so we set it manually after each injection.
    n_bits = p.bit_length()
    m_red = math.ceil(math.log2((p - 1) ** 2) + 1) - n_bits

    # Precompute constants for manual to_standard / to_montgomery.
    # conjugate(to_standard_qm) has a faulty-uncomputation bug under
    # boolean_simulation, so we apply the conversion manually.
    fwd_const = pow(2, m_red, p)  # to_standard multiplier
    rev_const = pow(2, -m_red, p)  # to_montgomery multiplier (inverse)

    if ctrl is None:
        anc[1] -= G[1]
    else:
        with control(ctrl):
            anc[1] -= G[1]  # step 2 //ctrl_sub_const_modp
    anc[0] -= G[0]  # step 1

    m = QuantumArray(qtype=QuantumBool(), shape=(2 * p.bit_length(),))
    l = QuantumModulus(p, inpl_adder=anc[0].inpl_adder)
    # step 3 & 4 & 6: compute (anc[1] * kaliski_inv(anc[0])) in standard form -> l
    with conjugate(kaliski_mod_inv)(anc[0], p, m):
        temp = QuantumModulus(p)
        injected_mul = temp << mul_func
        with conjugate(injected_mul)(anc[1], anc[0]):
            temp.m = -m_red
            temp *= fwd_const
            temp.m = 0
            cx(temp, l)  # copy standard value
            temp *= rev_const
            temp.m = -m_red
        temp.delete()
    m.delete()

    # step 5: anc[1] -= l * anc[0] (in standard form)
    temp = QuantumModulus(p)
    injected_mul = temp << mul_func
    with conjugate(injected_mul)(l, anc[0]):
        temp.m = -m_red
        temp *= fwd_const
        temp.m = 0
        anc[1] -= temp
        temp *= rev_const
        temp.m = -m_red
    temp.delete()

    if ctrl is None:
        anc[0] += 3 * G[0]
    else:
        with control(ctrl):
            anc[0] += 3 * G[0]  # step 9 //ctrl_add_const_modp

    # step 7 & 8: anc[0] -= l * l (in standard form)
    # Use a copy of l for the second factor because the injection mechanism
    # produces incorrect results when the same QuantumVariable is passed
    # for both arguments under boolean_simulation (JAX tracing).
    l_copy = QuantumModulus(p, inpl_adder=l.inpl_adder)
    cx(l, l_copy)
    temp = QuantumModulus(p)
    injected_mul = temp << mul_func
    with conjugate(injected_mul)(l, l_copy):
        temp.m = -m_red
        temp *= fwd_const
        temp.m = 0
        if ctrl is None:
            anc[0] -= temp  # step 8
        else:
            with control(ctrl):
                anc[0] -= temp  # step 8 //ctrl_sub_modp
        temp *= rev_const
        temp.m = -m_red
    temp.delete()  # step 10
    cx(l, l_copy)
    l_copy.delete()

    # step 11: anc[1] += l * anc[0] (in standard form)
    temp = QuantumModulus(p)
    injected_mul = temp << mul_func
    with conjugate(injected_mul)(l, anc[0]):
        temp.m = -m_red
        temp *= fwd_const
        temp.m = 0
        anc[1] += temp
        temp *= rev_const
        temp.m = -m_red
    temp.delete()

    # step 12-14: uncompute l via second kaliski inversion
    m = QuantumArray(qtype=QuantumBool(), shape=(2 * p.bit_length(),))
    with conjugate(kaliski_mod_inv)(anc[0], p, m):
        temp = QuantumModulus(p)
        injected_mul = temp << mul_func
        with conjugate(injected_mul)(anc[1], anc[0]):
            temp.m = -m_red
            temp *= fwd_const
            temp.m = 0
            cx(temp, l)  # XOR cancels: l ^= standard_temp, l becomes 0
            temp *= rev_const
            temp.m = -m_red
        temp.delete()
    m.delete()

    l.delete()
    if ctrl is None:
        anc[1] -= G[1]  # step 16
    else:
        with control(ctrl):
            anc[1] -= G[1]  # step 16 //ctrl_sub_const_modp

    if ctrl is None:
        inpl_rsub(anc[0], p)  # step 15
    else:
        with control(ctrl):
            inpl_rsub(anc[0], p)  # step 15 //ctrl_neg_modp

    anc[0] += G[0]  # step 17 (unconditional: undoes step 1)


def q_ec_mult_add(power, res, k, curve, n_bits=None):
    # Elliptic curve multiplication Q + kP
    #
    # n_bits must be supplied when running under JAX tracing
    # (e.g. @boolean_simulation) because k.size is a traced value
    # and the loop must be unrolled at trace time (each iteration
    # uses a different classical point doubling).
    if n_bits is not None:
        n = n_bits
    else:
        n = k.size
    p = curve.p
    merge(res[0], res[1], k)

    # Pre-compute all classical point doublings so that each
    # iteration uses the correct 2^i * P.
    powers = [tuple(power)]
    for _ in range(n - 1):
        powers.append(tuple(q_ec_double(list(powers[-1]), curve)))

    for i in range(n):
        with control(k[i]):
            q_ec_add_inpl(res, powers[i], p)
    return res
