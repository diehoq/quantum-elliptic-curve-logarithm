import src.quantum.ec_arithmetic as qECarithm
import pytest
import qrisp

primes = [
    3,
    5,
]  # 7, 11, 13, 17, 19, 23]


@pytest.mark.parametrize("v_cl", range(1, 23))
@pytest.mark.parametrize("p", primes)
def test_kaliski_quantum(v_cl, p):

    if v_cl >= p:
        pytest.skip()

    try:
        expected_inverse = pow(v_cl, -1, p)
    except ValueError:
        pytest.skip(f"No modular inverse exists for v={v_cl}, p={p} (non-coprime)")

    v = qrisp.QuantumModulus(p)
    v[:] = v_cl
    m = qrisp.QuantumArray(qtype=qrisp.QuantumBool(), shape=(2 * p.bit_length(),))

    quantum_inverse = qECarithm.kaliski_quantum(v, p, m)
    assert quantum_inverse.get_measurement() == {
        expected_inverse: 1
    }, f"Quantum inverse {quantum_inverse} did not match expected {expected_inverse} for v={v}, p={p}"
