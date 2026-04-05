# Quantum Elliptic Curve Discrete Logarithm — QDay Prize Submission

This repository contains a fully compilable implementation of **Shor's algorithm for the Elliptic Curve Discrete Logarithm Problem (ECDLP)**, targeting the [QDay Prize](https://www.qdayprize.org/) challenge. The implementation is built on [Qrisp](https://www.qrisp.eu/), a high-level quantum programming language, and demonstrates end-to-end compilation and resource estimation from 4-bit toy curves up to 256-bit (secp256k1-scale).

---

## Contact

- **Author**: Diego Polimeni
- **Email**: diego.polimeni.work@gmail.com
- **Website**: [diehoq.github.io](https://diehoq.github.io/)

## Background

I hold a Master's degree in Quantum Science and Engineering from EPFL (graduated June 2025). I am currently working as a researcher in Berlin, and I am a Contributor and Committer to the [Eclipse Foundation's Qrisp project](https://www.qrisp.eu/). My work focuses on quantum algorithm design and compilation, with a particular emphasis on quantum arithmetic and cryptographic applications.

## Key Broken

**TBD** — Resource estimation demonstrates compilation feasibility for both:

- all [QDay standard curves](https://www.qdayprize.org/curves) (4–21 bit), via [`resource_estimation_all_curves.ipynb`](resource_estimation_all_curves.ipynb)
- larger generated curves up to **256-bit**, via [`EC_discrete_log_compilation.ipynb`](EC_discrete_log_compilation.ipynb)

The algorithm derives private keys from the standard public keys using Shor's ECDLP algorithm, and the two notebooks separate the QDay-prize sweep from the larger-bit-size scaling study.

## Quantum Computer / Simulator

- **Model**: Qrisp built-in sparse-matrix simulator
- **Type**: Classical simulation of the quantum circuit
- **Access**: Local execution via the Qrisp framework (Python)

Since current quantum hardware cannot support circuits of this depth, the algorithm is validated through Qrisp's high-performance simulator. The compiled circuits are hardware-agnostic and can be exported to any gate-based quantum backend (IBM, Google, Rigetti, etc.) via Qrisp's backend interface.

## Algorithm Overview

The algorithm implements **Shor's algorithm for ECDLP** as described in:

> D. Polimeni, R. Seidel. *End-to-end compilable implementation of quantum elliptic curve logarithm in Qrisp*. [arXiv:2501.10228](https://arxiv.org/abs/2501.10228) (2025).

The paper describes a **static implementation** where the full quantum circuit for EC point arithmetic (addition, doubling, Kaliski modular inversion, Montgomery multiplication) is compiled ahead of time into a fixed gate sequence. All EC arithmetic operations are realized using Qrisp's `QuantumModulus` type and in-place modular arithmetic primitives.

### Dynamic Compilation via JAX and BigInteger Support

Beyond the static approach from the paper, this repository introduces **dynamic compilation through JAX** (Qrisp's `jasp` subsystem). Key advances:

- **JAX-traceable compilation**: The quantum algorithm is traced through JAX's just-in-time compilation (`jax.jit`), enabling dynamic circuit generation and optimization. This is powered by Qrisp's `jasp` module, which bridges quantum operations with JAX's transformation framework.
- **BigInteger support**: To handle prime fields exceeding the native 64-bit JAX integer limit, we use Qrisp's `BigInteger` class — a fixed-width array of 32-bit limbs registered as a JAX pytree node. This enables compilation for arbitrarily large primes (up to 256-bit and beyond) while remaining fully JAX-traceable. See [`BIGINTEGER_INTEGRATION.md`](BIGINTEGER_INTEGRATION.md) for the full technical summary.
- **`count_ops` resource estimation**: Qrisp's `@count_ops` decorator traces a single controlled EC point addition and extracts a complete gate-count breakdown without executing the full circuit. Combined with the structural insight that all $2n$ controlled additions have identical gate structure (only the classical point differs), this enables resource estimation at scale: $\text{total} = 2n \times \text{single\_add} + \text{overhead}$.

## Qrisp Dependencies

This project depends on two in-progress Pull Requests on the Qrisp repository (both will be merged soon):

1. **[PR #495](https://github.com/eclipse-qrisp/qrisp/pull/495)** — Core `jasp` (JAX-based quantum compilation) support and BigInteger arithmetic
2. **[PR #505](https://github.com/eclipse-qrisp/qrisp/pull/505)** — Additional fixes and enhancements for the EC logarithm compilation pipeline

Until these are merged into Qrisp's main branch, install the development branch specified in `requirements.txt`.

## Repository Structure

```
├── README.md                        # This file
├── brief.pdf                        # 2-page technical brief (QDay Prize requirement)
├── curves_and_keys.json             # QDay Prize standard curves (4–21 bit)
├── requirements.txt                 # Python dependencies (incl. Qrisp dev branch)
├── pyproject.toml                   # Project configuration
│
├── Shor_EC_discrete_log.ipynb       # Full Shor ECDLP algorithm (small example)
├── EC_discrete_log_compilation.ipynb # Compilation + resource estimation for larger curves up to 256-bit
├── resource_estimation_all_curves.ipynb # Compilation + resource estimation for all QDay standard curves
│
├── BIGINTEGER_INTEGRATION.md        # Technical notes on BigInteger integration
│
├── src/
│   ├── classical/
│   │   └── ec_arithmetic.py         # Classical EC arithmetic (reference implementation)
│   └── quantum/
│       └── ec_arithmetic.py         # Quantum EC arithmetic (Qrisp implementation)
│
├── tests/                           # Unit tests for all quantum primitives
│   ├── test_ec_addition.py
│   ├── test_ec_controlled_addition.py
│   ├── test_ec_doubling.py
│   ├── test_ec_mult_add.py
│   ├── test_kaliski_mod_inv.py
│   ├── test_proof_of_work.py
│   └── test_shor_ec_dlog.py
│
└── pytest-logs-*/                   # Test execution logs
```

## Instructions for Running

### Prerequisites

- Python 3.11+
- A virtual environment is recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/diehoq/quantum-elliptic-curve-logarithm.git
cd quantum-elliptic-curve-logarithm

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (includes Qrisp dev branch with PR #495 and #505)
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

### Running the Algorithm

**1. Full Shor ECDLP (small example):**
```bash
jupyter notebook Shor_EC_discrete_log.ipynb
```
This notebook runs the complete Shor algorithm on a small (5-bit) curve using Qrisp's simulator.

**2. Compilation and resource estimation for larger curves up to 256-bit:**
```bash
jupyter notebook EC_discrete_log_compilation.ipynb
```
This notebook is the large-curve scaling notebook. It focuses on compilation and resource estimation for representative generated curves, culminating in the 256-bit case.

**3. Resource estimation for ALL QDay standard curves:**
```bash
jupyter notebook resource_estimation_all_curves.ipynb
```
This notebook compiles and counts gates for every curve in the QDay standard set (4–21 bit). It produces the prize-wide resource table and scaling plots for the full challenge set.

**4. Run unit tests:**
```bash
pytest tests/ -v
```

### Notes

- Resource estimation for large bit sizes (128, 256) may take several minutes depending on hardware.
- The `@count_ops` approach traces a single EC addition and extracts gate counts without simulating the full circuit, making 256-bit estimation feasible on standard hardware.
- All curves use the equation $y^2 = x^3 + 7 \pmod{p}$ (matching secp256k1 form).
- The notebook split is intentional: [`resource_estimation_all_curves.ipynb`](resource_estimation_all_curves.ipynb) is for the QDay prize curve set, while [`EC_discrete_log_compilation.ipynb`](EC_discrete_log_compilation.ipynb) is for larger-curve compilation and scaling up to 256-bit.

## Other Documents

- [`BIGINTEGER_INTEGRATION.md`](BIGINTEGER_INTEGRATION.md) — Detailed technical documentation of the BigInteger integration across the project and Qrisp framework
- [`pytest-logs-*/`](pytest-logs-20260322-214828/) — Test execution logs from CI/local runs
- [`EC_discrete_log_compilation.ipynb`](EC_discrete_log_compilation.ipynb) — Compilation and resource estimation for larger curves up to 256-bit
- [`resource_estimation_all_curves.ipynb`](resource_estimation_all_curves.ipynb) — Compilation and resource estimation for all QDay standard curves

## License

See [LICENSE](LICENSE).
