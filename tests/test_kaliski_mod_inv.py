import src.quantum.ec_arithmetic as qECarithm
import pytest
import qrisp
from qrisp import measure, QuantumModulus, QuantumArray, QuantumBool,boolean_simulation, jaspify



primes = [3, 5, 7]  #11 , 13, 17, 19, 23]


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


@pytest.mark.parametrize("v_cl", range(1, 23))
@pytest.mark.parametrize("p", primes)
def test_kaliski_mod_inv_dynamic(v_cl, p):
    if v_cl >= p:
        pytest.skip()

    try:
        expected_inverse = pow(v_cl, -1, p)
    except ValueError:
        pytest.skip(f"No modular inverse exists for v={v_cl}, p={p} (non-coprime)")
    
    #@jaspify
    @boolean_simulation
    def main(v_cl):
        v = QuantumModulus(p)
        v[:] = v_cl
        m = QuantumArray(qtype=QuantumBool(), shape=(2 * p.bit_length(),))
        qECarithm.kaliski_mod_inv(v, p, m)
        return measure(v)
    
    result = main(v_cl)
    assert result == expected_inverse, f"Quantum inverse {result} did not match expected {expected_inverse} for v={v_cl}, p={p}"