import src.quantum.ec_arithmetic as qECarithm
import pytest
import qrisp
from qrisp import measure, QuantumModulus, QuantumArray, QuantumBool, boolean_simulation
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger



primes = [3, 5, 7, 11, 13, 17, 19, 23]
BI_SIZE = 1
BIGINT_PRIMES = [7, 13]

@pytest.mark.parametrize("v_cl", range(1, 23))
@pytest.mark.parametrize("p", primes)
def test_kaliski_mod_inv(v_cl, p):
    if v_cl >= p:
        pytest.skip()

    try:
        expected_inverse = pow(v_cl, -1, p)
    except ValueError:
        pytest.skip(f"No modular inverse exists for v={v_cl}, p={p} (non-coprime)")

    v = qrisp.QuantumModulus(p)
    v[:] = v_cl
    m = qrisp.QuantumArray(qtype=qrisp.QuantumBool(), shape=(2 * p.bit_length(),))

    qECarithm.kaliski_mod_inv(v, p, m)
    assert v.get_measurement() == {
        expected_inverse: 1
    }, f"Quantum inverse did not match expected {expected_inverse} for v_cl={v_cl}, p={p}"

@pytest.mark.parametrize("p", primes)
def test_kaliski_mod_inv_dynamic(p):
    @boolean_simulation
    def main(v_cl):
        v = QuantumModulus(p)
        v[:] = v_cl
        m = QuantumArray(qtype=QuantumBool(), shape=(2 * p.bit_length(),))
        qECarithm.kaliski_mod_inv(v, p, m)
        return measure(v)

    for v_cl in range(1, 23):
        if v_cl >= p:
            continue

        try:
            expected_inverse = pow(v_cl, -1, p)
        except ValueError:
            continue

        result = main(v_cl)
        assert result == expected_inverse, (
            f"Quantum inverse {result} did not match expected {expected_inverse} "
            f"for v={v_cl}, p={p}"
        )


@pytest.mark.parametrize("p", primes[:5])
def test_kaliski_mod_inv_superposition(p):
    """Prepare an equal superposition over all allowed inputs and verify
    that modular inversion preserves the uniform distribution."""
    v = qrisp.QuantumModulus(p)
    m = qrisp.QuantumArray(qtype=qrisp.QuantumBool(), shape=(2 * p.bit_length(),))

    allowed_inputs = range(1, p)
    v[:] = {value: 1 for value in allowed_inputs}

    qECarithm.kaliski_mod_inv(v, p, m)

    results = v.get_measurement()
    expected_outputs = {pow(value, -1, p) for value in allowed_inputs}
    expected_probability = 1 / (p - 1)

    assert set(results.keys()) == expected_outputs, (
        f"p={p}: outcomes {set(results.keys())} != expected inverses {expected_outputs}"
    )
    for value in expected_outputs:
        assert results[value] == pytest.approx(expected_probability, abs=1e-6), (
            f"p={p}: output {value} had probability {results[value]}, "
            f"expected {expected_probability}"
        )


@pytest.mark.parametrize("p", BIGINT_PRIMES)
def test_kaliski_mod_inv_biginteger_dynamic(p):
    p_bi = BigInteger.create_static(p, BI_SIZE)

    @boolean_simulation
    def main(v_cl):
        v = QuantumModulus(p_bi)
        v[:] = v_cl
        m = QuantumArray(qtype=QuantumBool(), shape=(2 * p.bit_length(),))
        qECarithm.kaliski_mod_inv(v, p, m)
        return measure(v)

    for v_cl in range(1, p):
        try:
            expected_inverse = pow(v_cl, -1, p)
        except ValueError:
            continue

        result = main(v_cl)
        result_int = int(result()) if isinstance(result, BigInteger) else int(result)
        assert result_int == expected_inverse, (
            f"BigInteger inverse {result_int} did not match expected {expected_inverse} "
            f"for v={v_cl}, p={p}"
        )
