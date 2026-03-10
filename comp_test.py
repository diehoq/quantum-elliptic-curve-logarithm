import qrisp
import numpy as np
import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm

def benchmark_circuit(res):

    qubit_count = len([qb for qb in res.qs.qubits if qb.allocated])

    compiled_circuit = res.qs.compile()

    gate_counts = compiled_circuit.transpile().count_ops()
    print(gate_counts)
    cnot_count = gate_counts.get('cx', 0)
    t_count = gate_counts.get('t', 0)
    t_depth = compiled_circuit.depth()  # Assumes depth of T gates aligns with circuit depth

    return {
        "qubit_count": qubit_count,
        "t_count": t_count,
        "cnot_count": cnot_count,
        "t_depth": t_depth
    }

p=7
a=5
b=4
curve = clECarithm.EllCurve(a, b, p)

x1 = qrisp.QuantumModulus(2**(p.bit_length()))
x1[:] = 1
mod_p = qrisp.QuantumModulus(p, inpl_adder=qrisp.gidney_adder)
anc = qrisp.QuantumArray(qtype=mod_p, shape=(2,))
anc[:] = [2,6]

G = [0,5]

res_ec_add = qECarithm.q_ec_add_inpl(anc, G, p)

bm = benchmark_circuit(res_ec_add[0])
print(bm)
