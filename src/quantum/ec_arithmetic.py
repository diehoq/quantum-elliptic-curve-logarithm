import contextlib
import functools
import math
import numpy as np
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
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
    modinv as bi_modinv,
    smallest_power_of_two,
)


def _p_int(p):
    """Return p as a Python int (works for both int and BigInteger).

    Safe to call inside jax.jit ONLY when p's digits are concrete (non-traced).
    This works for BigIntegers from QuantumVariable aux_data (e.g., v.modulus)
    but NOT for BigIntegers created inside jit.
    """
    if isinstance(p, BigInteger):
        digits = np.asarray(p.digits)
        result = 0
        for i in range(len(digits)):
            d = int(digits[i])
            if d:
                result += d << (32 * i)
        return result
    return int(p)


def _bi(val, p):
    """Convert val to BigInteger if p is BigInteger, else return val unchanged."""
    if isinstance(p, BigInteger):
        return BigInteger.create_static(int(val), p.digits.shape[0])
    return val


def to_montgomery(x, p_int):
    """Convert x to Montgomery form. p_int must be a Python int."""
    n = p_int.bit_length()
    # QuantumModulus __imul__ handles int→BigInteger via encoder automatically
    const = pow(2, n, p_int)
    x *= const
    return x


def to_standard(x, p_int):
    """Convert x from Montgomery form. p_int must be a Python int."""
    n = p_int.bit_length()
    const = pow(pow(2, n, p_int), -1, p_int)
    x *= const
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
def inpl_rsub(r, p_int, mod_int=None):
    # Compute r := (p - r) mod p.
    # p_int must be a Python int.
    # mod_int: optional Python int for r's modulus (avoids _p_int on traced BigInteger).
    # The bit-trick (NOT + add modulus+1 + modular-add p) creates intermediate
    # value == modulus when r=0, which breaks mod_adder (its ancilla can't
    # uncompute because a=0 makes forward/reverse additions all no-ops).
    # This only matters when p % r.modulus == 0 (i.e. modulus == p), because
    # then __iadd__ pre-reduces p to 0 and mod_adder receives a=0.
    # For modulus == 2p, the add is non-degenerate and mod_adder works fine.
    use_bi = isinstance(r.modulus, BigInteger)
    bi_size = r.modulus.digits.shape[0] if use_bi else None

    def _coerce_const(value):
        """Route int→BigInteger via BigInteger.coerce when in BigInteger mode."""
        if use_bi:
            return BigInteger.coerce(int(value), bi_size)
        return value

    mod_i = mod_int if mod_int is not None else _p_int(r.modulus)
    mod_plus_1 = _coerce_const(mod_i + 1)
    p_value = _coerce_const(p_int)
    if p_int % mod_i == 0:
        # r += p reduces to mod_adder(0,...) which faults on value=modulus.
        # Fix: skip when r=0 (it's a no-op since (p-0) mod p = 0).
        is_zero = QuantumBool()
        eq = is_zero << (lambda a, b: a == b)
        with conjugate(eq)(r, 0):
            with control(is_zero, ctrl_state="0"):
                x(r)
                r.inpl_adder(mod_plus_1, r)
                r += p_value
        is_zero.delete()
    else:
        x(r)
        r.inpl_adder(mod_plus_1, r)
        r += p_value


# Consider moving this function to qrisp source code
# This function is used to compute the modular inverse of a number
@qache(static_argnums=[1])
def kaliski_mod_inv(v, p, m):
    # p is always a Python int (required for @qache hashing).
    # Detect BigInteger mode from the quantum register's modulus.
    use_bi = isinstance(v.modulus, BigInteger)
    bi_size = v.modulus.digits.shape[0] if use_bi else None

    def _mk_bi(val):
        """Convert int val to BigInteger if v uses BigInteger modulus."""
        if use_bi:
            return BigInteger.create_static(int(val), bi_size)
        return val

    def _load_qfloat_constant(qf, value, width):
        """Load a non-negative Python int without going through float64 encoding."""
        for bit in range(width):
            if (value >> bit) & 1:
                x(qf[bit])

    def _toggle_if_qfloat_equals_int(qf, value, target, width):
        chunk_size = 8
        chunk_flags = []

        for start in range(0, width, chunk_size):
            stop = min(start + chunk_size, width)
            flag = QuantumBool()
            ctrl_state = "".join(
                "1" if (value >> bit) & 1 else "0" for bit in range(start, stop)
            )
            mcx([qf[bit] for bit in range(start, stop)], flag, ctrl_state=ctrl_state)
            chunk_flags.append((flag, start, stop, ctrl_state))

        reduction_gates = []
        current_flags = [flag for flag, _, _, _ in chunk_flags]
        while len(current_flags) > 1:
            next_flags = []
            for i in range(0, len(current_flags), 2):
                if i + 1 == len(current_flags):
                    next_flags.append(current_flags[i])
                    continue
                merged_flag = QuantumBool()
                mcx([current_flags[i], current_flags[i + 1]], merged_flag)
                reduction_gates.append((merged_flag, current_flags[i], current_flags[i + 1]))
                next_flags.append(merged_flag)
            current_flags = next_flags

        cx(current_flags[0], target)

        for merged_flag, left_flag, right_flag in reversed(reduction_gates):
            mcx([left_flag, right_flag], merged_flag)
            merged_flag.delete()

        for flag, start, stop, ctrl_state in reversed(chunk_flags):
            mcx([qf[bit] for bit in range(start, stop)], flag, ctrl_state=ctrl_state)
            flag.delete()

    n = p.bit_length()
    # For QuantumModulus construction, we need BigInteger modulus when use_bi.
    # For quantum operations (+=, -=, >, *=), we pass Python ints directly;
    # the QuantumModulus encoder handles int→BigInteger conversion automatically.
    two_p = _mk_bi(2 * p)
    # Convert to Montgomery — pass Python int p for classical computation
    to_montgomery(v, p)
    u = QuantumFloat(n)
    _load_qfloat_constant(u, p, n)
    r = QuantumModulus(two_p, inpl_adder=v.inpl_adder)
    r[:] = 0
    s = QuantumModulus(two_p, inpl_adder=v.inpl_adder)
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

        larger = r > _mk_bi(p)
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
    _toggle_if_qfloat_equals_int(u, p, v_was_zero, n)
    with control(v_was_zero):
        r += p
    _toggle_if_qfloat_equals_int(u, p, v_was_zero, n)
    v_was_zero.delete()

    inpl_rsub(r, p, mod_int=2 * p)

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
    p_i = _p_int(p)
    s = ((3 * (P[0] * P[0] % p_i) + curve.a) % p_i) * pow((2 * P[1]) % p_i, -1, p_i)
    xr = (s * s - 2 * P[0]) % p_i
    yr = P[1] - s * ((P[0] - xr) % p_i) % p_i
    # CHOOSE APPROPRIATE RETURN TYPE
    return [xr, (p_i - yr) % p_i]


def montgomery_reduction_shift(p_int):
    """Return the reduction shift ``m`` of Qrisp's quantum-quantum Montgomery product.

    Multiplying two ``QuantumModulus`` registers yields ``a * b * 2**(-m)``
    with ``m = ceil(log2((p - 1)**2)) + 1 - ceil(log2(p))``. Qrisp computes this
    in ``montgomery_mod_mul`` (static) and ``qq_montgomery_multiply_modulus``
    (jasp). The formula is duplicated here because the injection operator
    (``<<``) does not propagate the product's ``.m``. If the two ever drift
    apart, every product is off by a constant power of two;
    ``tests/test_montgomery_product.py`` guards against that.
    """

    # Exact integer ceil(log2(x)), matching Qrisp's smallest_power_of_two
    # (floats lose precision for 256-bit primes).
    def ceil_log2(x):
        return (x - 1).bit_length() if x > 1 else 0

    return ceil_log2((p_int - 1) ** 2) + 1 - ceil_log2(p_int)


def q_ec_add_inpl(anc, G, p, ctrl=None):
    """In place, (anc[0], anc[1]) := (anc[0], anc[1]) + G, optionally controlled.

    ``G`` is a classical point. ``p`` is the field prime as an int or a
    BigInteger with concrete digits. Exceptional cases (anc = ±G, or
    anc + G = ±G) are not handled; see the affine-addition literature.
    """
    # Qrisp traces QuantumModulus.modulus inside qached functions, so resolve
    # the prime to a concrete Python int here. It reaches the kernel through a
    # closure rather than a static argument: custom_control traces the
    # controlled version without honouring static_argnames.
    kernel = _ec_add_kernel(_p_int(p))
    kwargs = {} if ctrl is None else {"ctrl": ctrl}
    return kernel(anc, G, **kwargs)


@functools.cache
def _ec_add_kernel(p_int):
    """Build (once per prime) the custom-controlled in-place point addition."""

    @custom_control
    def kernel(anc, G, ctrl=None):
        _q_ec_add_inpl_body(anc, G, p_int, ctrl)

    return kernel


@contextlib.contextmanager
def standard_form_product(a, b, p_mod, p_int):
    """Temporarily hold ``a * b mod p`` in standard form; uncomputed on exit.

    Qrisp's quantum-quantum product returns ``a * b * 2**(-m)`` (Montgomery
    form). Inside the ``with`` block the yielded register holds the plain
    product, so it can be XOR-copied or added into other registers. ``p_mod``
    is the modulus for the temporary register (int or BigInteger matching the
    operands), ``p_int`` the same prime as a Python int.
    """
    m_red = montgomery_reduction_shift(p_int)
    # Apply the to_standard / to_montgomery conversions manually:
    # conjugate(to_standard_qm) has a faulty-uncomputation bug under
    # boolean_simulation. The int constants are encoded to BigInteger by
    # QuantumModulus when needed.
    fwd_const = pow(2, m_red, p_int)
    rev_const = pow(2, -m_red, p_int)

    temp = QuantumModulus(p_mod)
    injected_mul = temp << (lambda x, y: x * y)
    with conjugate(injected_mul)(a, b):
        # Injection (<<) doesn't propagate .m from the product, so set it.
        temp.m = -m_red
        temp *= fwd_const
        temp.m = 0
        yield temp
        temp *= rev_const
        temp.m = -m_red
    temp.delete()


def _q_ec_add_inpl_body(anc, G, p_int, ctrl):
    # Following C3 in the paper.
    # Build new registers from the concrete prime (an int, or a BigInteger with the
    # same limb count as anc's modulus) instead of the traced anc[0].modulus.
    p_i = p_int
    p_raw = _bi(p_i, anc[0].modulus)
    n_bits = p_i.bit_length()

    def controlled(op):
        if ctrl is None:
            op()
        else:
            with control(ctrl):
                op()

    # G values are classical constants; QuantumModulus encodes them to
    # BigInteger when the modulus is a BigInteger.
    def sub_g1():
        anc[1] -= G[1]

    controlled(sub_g1)  # step 2 //ctrl_sub_const_modp
    anc[0] -= G[0]  # step 1

    # step 3 & 4 & 6: l = anc[1] * kaliski_inv(anc[0]) in standard form
    m = QuantumArray(qtype=QuantumBool(), shape=(2 * n_bits,))
    l = QuantumModulus(p_raw, inpl_adder=anc[0].inpl_adder)
    with conjugate(kaliski_mod_inv)(anc[0], p_i, m):
        with standard_form_product(anc[1], anc[0], p_raw, p_i) as temp:
            cx(temp, l)
    m.delete()

    # step 5: anc[1] -= l * anc[0]
    with standard_form_product(l, anc[0], p_raw, p_i) as temp:
        anc[1] -= temp

    def add_3g0():
        anc[0] += G[0] * 3

    controlled(add_3g0)  # step 9 //ctrl_add_const_modp

    # step 7 & 8: anc[0] -= l * l
    # Use a copy of l for the second factor: Qrisp's quantum-quantum product
    # does not support passing the same register as both operands.
    l_copy = QuantumModulus(p_raw, inpl_adder=l.inpl_adder)
    cx(l, l_copy)
    with standard_form_product(l, l_copy, p_raw, p_i) as temp:

        def sub_l_squared():
            anc[0] -= temp

        controlled(sub_l_squared)  # step 8 //ctrl_sub_modp
    cx(l, l_copy)  # step 10
    l_copy.delete()

    # step 11: anc[1] += l * anc[0]
    with standard_form_product(l, anc[0], p_raw, p_i) as temp:
        anc[1] += temp

    # step 12-14: uncompute l via a second Kaliski inversion
    m = QuantumArray(qtype=QuantumBool(), shape=(2 * n_bits,))
    with conjugate(kaliski_mod_inv)(anc[0], p_i, m):
        with standard_form_product(anc[1], anc[0], p_raw, p_i) as temp:
            cx(temp, l)  # XOR cancels: l becomes 0
    m.delete()
    l.delete()

    controlled(sub_g1)  # step 16 //ctrl_sub_const_modp

    def negate_x():
        inpl_rsub(anc[0], p_i, mod_int=p_i)

    controlled(negate_x)  # step 15 //ctrl_neg_modp
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
    # Under JAX tracing all QVs share a single TracingQuantumSession,
    # so merge is a no-op.  Calling it would trigger __iter__ on dynamic QVs.
    from qrisp.jasp import check_for_tracing_mode
    if not check_for_tracing_mode():
        merge(res[0], res[1], k)

    # Pre-compute all classical point doublings so that each
    # iteration uses the correct 2^i * P.
    powers = [tuple(power)]
    for _ in range(n - 1):
        powers.append(tuple(q_ec_double(list(powers[-1]), curve)))

    for i in range(n):
        # When modulus is BigInteger, convert classical coordinates to
        # BigIntegers so they don't overflow JAX's int64 during tracing.
        if isinstance(p, BigInteger):
            bi_size = p.digits.shape[0]
            g_i = (BigInteger.create_static(int(powers[i][0]), bi_size),
                   BigInteger.create_static(int(powers[i][1]), bi_size))
        else:
            g_i = powers[i]
        q_ec_add_inpl(res, g_i, p, ctrl=k[i])
    return res
