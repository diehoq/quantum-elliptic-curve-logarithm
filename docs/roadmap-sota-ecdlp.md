# Roadmap: a state-of-the-art ECDLP implementation in Qrisp

Status: proposal, November 2026. Audience: contributors to this repository and the Qrisp arithmetic maintainers.

This document lays out how to move this repository from the design of Polimeni & Seidel ([arXiv:2501.10228](https://arxiv.org/abs/2501.10228)) to the design used by the 2026 resource estimates (Google, Schrottenloher, IonQ, Litinski). Those designs use windowed point additions with table lookups, inversion-free division, measurement-based uncomputation, and a single recycled control register. The roadmap also accounts for the arithmetic refactor currently under way in Qrisp.

## 1. Where we are

The current implementation, in [`src/quantum/ec_arithmetic.py`](../src/quantum/ec_arithmetic.py):

| Aspect | Current design |
|---|---|
| Scalar multiplication | Double-and-add. Each exponent qubit controls one addition of a classical point `2^i·P` (`q_ec_mult_add`), so there are 2n controlled additions for the two exponent registers. |
| Point addition | Affine, in place, quantum accumulator plus a **classical** point (`q_ec_add_inpl`, circuit C3 of the paper). |
| Division | Two Kaliski inversions per addition (compute λ, then uncompute λ), plus 4 Montgomery products. |
| Field arithmetic | Qrisp `QuantumModulus` (Montgomery representation) with Gidney adders. |
| Phase estimation | Two full (n+1)-qubit exponent registers and a coherent inverse QFT. |
| Exceptional cases | Not handled (P = ±G, or P + G = ±G). |
| Toolchain | JAX-traced (jasp) compilation, `BigInteger` primes up to 256 bits, and `count_ops` resource estimation. |

Cost is dominated by the 4n Kaliski inversions. Litinski measures one Kaliski inversion at 26n² + 2n Toffoli gates, more than ten times a multiplication. Qubit count is dominated by the two 2n-qubit exponent registers and the inversion workspace.

The dynamic-mode bugs fixed in #23 (Montgomery-shift drift, traced modulus) showed that our arithmetic is tightly coupled to Qrisp internals that are changing. The roadmap therefore puts an explicit boundary between our code and Qrisp's arithmetic.

## 2. Target design

As of 2026, every leading design follows the same template. Google ([arXiv:2603.28846](https://arxiv.org/abs/2603.28846)) gives two variants for secp256k1: ≤1200 logical qubits with ≤90M Toffoli, or ≤1450 qubits with ≤70M Toffoli, with circuits withheld. Schrottenloher ([arXiv:2606.02235](https://arxiv.org/abs/2606.02235)) publishes an open circuit that matches it to within about 1.5% in qubits and 6.5–10% in Toffoli. IonQ (ePrint 2026/1916) optimizes the open circuit further.

1. **Windowed lookup–add–unlookup.**
   - Both exponents are split into windows of w bits, giving 2n/w windows in total.
   - Per window j, a QROM loads the precomputed affine point `T_j[k] = k·2^(wj)·P` (or `·Q`) into a quantum register, keyed by the w window qubits.
   - A **quantum-quantum** affine point addition adds it to the accumulator.
   - The lookup is then erased by measurement-based unlookup. Measure the target in the X basis, then apply a phase fix-up lookup costing about 2√L Toffoli (Gidney, [arXiv:1905.07682](https://arxiv.org/abs/1905.07682)).
2. **Division without a separate inversion.**
   - λ = (y₂ − y₁)/(x₂ − x₁) comes from a constant-iteration Euclid "dialog" with Bézout reconstruction: about 1.413n + 2.4√n iterations. The same reconstruction also gives the in-place multiplication for free.
   - Sources: Schrottenloher 2026, after Khattar et al. [arXiv:2510.10967](https://arxiv.org/abs/2510.10967).
   - Kaliski remains the simple, well-understood fallback.
3. **Standard (non-Montgomery) representation.**
   - Use special-form reduction where the prime allows it, e.g. pseudo-Mersenne for secp256k1.
   - Use approximate comparisons that check only the top 40–50 bits.
4. **Measurement-based uncomputation everywhere.** This covers AND gates (already used by Qrisp's Gidney adder), lookups, and comparison flags.
5. **One recycled w-qubit control register with a semiclassical QFT.**
   - After each window: apply the classically controlled phase corrections, do a w-qubit QFT, measure, and reset.
   - Recover the dropped bits with Ekerå's post-processing ([arXiv:1905.09084](https://arxiv.org/abs/1905.09084)). The first window is a direct lookup. The last three windows are dropped and recovered by classical search, leaving 2n/w − 4 windowed additions.
6. **Exceptional cases.** For cryptographic sizes they are ignored, because their probability is about (number of additions)/r. For toy curves they must be handled, either with exact flags (Litinski [arXiv:2306.08585](https://arxiv.org/abs/2306.08585); Garn–Kan [arXiv:2503.02984](https://arxiv.org/abs/2503.02984)) or with a random starting offset.

Cost model, to leading order:

```
Toffoli ≈ (PA + 3·2^w) · (2n/w − 4)        qubits ≈ PA_qubits + w
```

Here PA is the cost of one quantum-quantum point addition. For n = 256 and w = 16, this means 28 additions of roughly 2M Toffoli each. ECDSA.Fail ([arXiv:2609.09582](https://arxiv.org/abs/2609.09582)) reports 1.68M average Toffoli per windowed addition at 1,162 qubits.

The knobs, roughly in order of impact:

1. Division algorithm.
2. Window size w.
3. Exact vs. approximate arithmetic.
4. Uncomputation strategy (measurement-based vs. mirrored, about 2×).
5. Exponent truncation and post-processing.
6. Lookup construction. This is minor at w = 16.

## 3. What Qrisp gives us, and what is moving

The following is checked against upstream `main` @ `2ba49e8` (2026-10-28) and the open PRs and issues.

**Usable today**
- `gidney_adder` and `cuccaro_adder`. Both support quantum-quantum and classical-quantum addition, both are controlled, and both work in static, jasp and boolean simulation.
- `mod_adder(a, b, inpl_adder, N, ctrl)`.
- QuantumModulus multiplication and in-place multiplication.
- `q_switch(..., method="tree")`, a Khattar–Gidney style multiplexer that also works in jasp.
- `int_encoder`, which XORs a classical int or BigInteger into a register.
- Mid-circuit `measure` and `reset` in jasp, and classical control via `q_cond`.
- `count_ops`, `depth` and `num_qubits` profiling in jasp.

**Missing**
- No QROM, unary-iteration, select-swap or unlookup primitive. These are tracked in Qrisp #845 and #849, with no code yet.
- No quantum modular inversion or division.
- No semiclassical-QFT helper for this setting.
- No T-count metric; `t` and `t_dg` have to be summed by hand.

**Moving**

Qrisp Discussion #944 puts the arithmetic refactor at "now + next year", with fault-tolerant primitives after it:

| Change | Impact on us |
|---|---|
| #802 and #958: `QuantumModulus.modulus` and `.m` are now traced | Already handled in #23 by resolving the prime up front. |
| #1007 (open): `m` static unless traced | Setting `.m` on a qached argument would raise. Our helper sets `.m` only on fresh registers, which is safe. |
| #933 and #883 (open): custom-control and inverse capture bugs | Possible silently wrong results with `custom_control`. Keep exhaustive tests. |
| #1001 and #990 (open): adder refactors, with breaking signature changes | Only touch adders through one wrapper module. |
| #965: `jasp_mod_adder` removed | Use `mod_adder`. |
| #468: injection (`<<`) drops `.m` | Keep `montgomery_reduction_shift`, guarded by `tests/test_montgomery_product.py`. |
| #919: `BigInteger.get_bit` has no bounds guard | Allocate one spare limb for 256-bit primes. |

Found during #23, and worth reporting upstream: `custom_control` forwards `static_argnames` to `qache`, but the controlled-version trace (`make_jaspr(..., **cusc_kwargs)`) ignores them. Static arguments therefore become tracers.

Simulation constraints:
- `boolean_simulation` treats phase gates as no-ops and raises on `H`. So arithmetic is validated with boolean simulation, while measurement-based unlookup and the semiclassical QFT need a reversible fallback, or the sparse statevector simulator on toy curves.
- Use `callback_threshold=500` to keep XLA compile times manageable.

**Strategy.** Build the missing fault-tolerant primitives here first, behind small interfaces designed to be upstreamed into Qrisp #845 and #849. Wrap Qrisp's arithmetic in one adapter module, so each Qrisp bump is a one-file change plus the exhaustive tests. Pin Qrisp by commit SHA and bump deliberately, as we do now.

## 4. Phases

Each phase ends with exhaustive correctness tests on small primes under boolean simulation, plus a Toffoli and qubit table produced by `count_ops`. Sizes are rough (S ≈ 1–2 weeks, M ≈ 3–5 weeks, L ≈ 6–10 weeks for one person).

### Phase 0: Baseline and hygiene (S, largely done)
- ✅ Qrisp pinned to upstream `main`; dynamic-mode failures fixed (#23); ruff and lint CI (#24).
- ✅ Exhaustive regression test for the Montgomery product shift.
- ☐ Fix README instructions that do not work. For example, `pip install -e .` fails: `pyproject.toml` has no `[project]` table, so setuptools cannot infer a valid distribution name.
- ☐ Add exceptional-case tests that document today's behaviour: (2, 6) + G = −G on p = 7 reports faulty uncomputation.
- ☐ Report the `custom_control` `static_argnames` bug upstream.

### Phase 1: Cost model and measurement harness (S)
- A pure-Python cost model of the target algorithm. Inputs: n, w, division method, adder, and uncompute strategy. Output: Toffoli and qubit counts. Calibrate it against the published formulas: Litinski's Kaliski cost of 26n² + 2n, Roetteler et al. ([arXiv:1706.06752](https://arxiv.org/abs/1706.06752)), and Häner et al. ([arXiv:2001.09580](https://arxiv.org/abs/2001.09580)).
- A `count_ops`-based harness that sums Toffoli or T gates and peak qubits for any traced kernel, from n = 4 to 256, with a CSV export. It replaces the ad-hoc `count_ops_repro.py` and the notebook code.
- A **classical simulator of Shor's output distribution** for known d and r, following Ekerå. It lets us test the post-processing (dropped windows, limited search) without quantum simulation.
- *Exit:* a baseline table for the current design on the QDay curves (4–21 bits) and synthetic curves up to 256 bits, and agreement between the cost model and `count_ops` within 10%.

### Phase 2: Field layer behind an interface (M)
- A `FieldOps` protocol covering:
  - add, sub and negate, each in quantum-quantum, classical-quantum and controlled forms;
  - `mul_into` (out-of-place) and `mul_inplace_by_const`;
  - `square_into`;
  - comparison flags.
- Two backends:
  - `QrispMontgomeryField`, today's code wrapped, using `standard_form_product`.
  - `StandardField`, built on `QuantumFloat` with Gidney adders. It offers schoolbook or Karatsuba multiplication (see Gidney's quantum schoolbook multiplication with fewer Toffoli gates, [arXiv:2410.00899](https://arxiv.org/abs/2410.00899)), Barrett or special-form reduction, and an optional approximate comparison mode.
- Squaring must not alias its operand; Qrisp's quantum-quantum multiplication does not support that yet. Use a copy, as today, until there is a dedicated squaring circuit.
- *Exit:* both backends pass exhaustive tests for p ∈ {7, 13, 17, 31} in controlled and uncontrolled forms, and `q_ec_add_inpl` runs on either backend.

### Phase 3: Lookup primitives (M)
- `qrom_lookup(index, table, target, ctrl=None)` built on `q_switch(method="tree")` and `int_encoder`. The table is a `jnp` uint32 array of shape (L, limbs), which works with BigInteger-sized words.
- `qrom_unlookup` in two variants:
  - reversible (mirrored lookup), so it runs under boolean simulation;
  - measurement-based (X-basis measurement plus a phase fix-up lookup of size √L), for statevector simulation and resource counts.
- Later, optionally: select-swap / QROAM for qubit–Toffoli trade-offs (Low–Kliuchnikov–Schaeffer, [arXiv:1812.00954](https://arxiv.org/abs/1812.00954); Berry et al., [arXiv:1902.02134](https://arxiv.org/abs/1902.02134)), and low-ancilla unary iteration (Khattar–Gidney, [arXiv:2407.17966](https://arxiv.org/abs/2407.17966)).
- *Exit:*
  - Correctness for random tables up to L = 2^10.
  - Measured lookup cost within a small constant of L − 1 Toffoli, and unlookup cost about 2√L.
  - An upstream PR or design note for Qrisp #845.

### Phase 4: Quantum-quantum affine point addition (L)
- `q_ec_add_qq(acc, pt, field, divider)` adds a quantum point to the accumulator in place. This is the core change, because today's addition takes only a classical point.
- Divider 1 is Kaliski, reused and generalised to quantum-quantum operands: compute and uncompute λ, mirroring circuit C3.
- Divider 2 is Euclid dialog with Bézout reconstruction (Schrottenloher 2026, with open code at gitlab.inria.fr/capsule/qarton-projects/ec-point-addition). It gives division and the in-place multiply in one pass.
- Exceptional-case handling sits behind a flag:
  - `exceptional="ignore"` for large curves;
  - `exceptional="flags"` for toy curves: four flags (x₁ = x₂, y₁ = −y₂, P₁ = O, P₂ = O), with O represented as (0, 0) and the doubling slope taken from the table, as in Litinski and Garn–Kan.
- *Exit:*
  - Exhaustive tests over all point pairs on p = 13 and p = 43 (the QDay 4- and 6-bit curves) for both dividers.
  - Toffoli per addition within 25% of the published formulas.
  - Divider 2 beats divider 1 from n ≥ 32 in the harness.

### Phase 5: Windowed scalar multiplication and semiclassical Shor (L)
- `q_ec_windowed_mult_add(acc, P, Q, w)`: the first window is a direct lookup, then (2n/w − 4) rounds of lookup–add–unlookup, alternating between the P and Q tables.
- A recycled w-qubit control register: `h` → window body → classically controlled phase corrections → w-qubit QFT → `measure` → `reset`, all with jasp mid-circuit measurement and `q_cond`.
- Post-processing (Ekerå): recover the dropped windows by limited search and handle failed runs. Validate it first with the Phase 1 classical simulator.
- *Exit:*
  - End-to-end private-key recovery on the QDay 4- to 8-bit curves with the sparse statevector simulator, matching `curves_and_keys.json`.
  - Measured success probability consistent with the classical simulator.
  - `test_shor_ec_dlog.py` rewritten around the new driver.

### Phase 6: Resource estimation at scale (M)
- Sweep w, divider, adder, and uncomputation strategy for n ∈ {64, 128, 192, 224, 256}, on secp256k1 and P-256 (generic-prime reduction).
- Compare with Google, Schrottenloher, IonQ, Litinski, and Häner et al.
- Replace the notebooks' resource sections with the harness, and refresh `brief.tex`.
- *Exit:* a reproducible table (script plus CSV in CI artifacts) and a write-up of where we land, and why.

### Phase 7: Research extensions (open-ended)
- Low-qubit variants:
  - register sharing (Proos–Zalka, quant-ph/0301141);
  - Luo et al. 2026 ([arXiv:2607.13816](https://arxiv.org/abs/2607.13816)): 835 logical qubits at about 2·10⁹ Toffoli;
  - Chevignard–Fouque–Schrottenloher (ePrint 2026/280), about 3.12n qubits. Its n = 256 figure is inconsistent across versions (1,193 vs. 1,098); check the PDF before citing.
- Batched division, using Montgomery's trick across parallel instances (Litinski).
- Formal or fuzz-style correctness evidence for large-n kernels, analogous to Google's 9,024-input test argument.

## 5. Sequencing against the Qrisp refactor

```
2026 Q4   Phase 0 ✔, Phase 1, Phase 2 (adapter isolates Qrisp churn)
2027 Q1   Phase 3 (lookups, upstream design to Qrisp #845/#849), Phase 4 with Kaliski
2027 Q2   Phase 4 with Euclid dialog, Phase 5
2027 Q3   Phase 6; re-target Phase 3 onto Qrisp-native lookups if they have landed
```

On every Qrisp bump:
- run the exhaustive field tests (Phase 2);
- run the lookup tests (Phase 3);
- run the point-addition tests (Phase 4);
- diff the resource table.

A changed Toffoli count must be explained in the bump PR.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Qrisp API churn (traced attributes, adder signatures, custom-control bugs) | Pin by SHA, keep one adapter module, and run exhaustive small-prime tests in CI on every bump. |
| Measurement-based steps cannot run under boolean simulation | Provide a reversible variant of every measurement-based step, and validate phases on toy curves with the statevector simulator. |
| Exceptional cases dominate on toy curves | Exact-flag mode (Phase 4), and a random starting offset for the scalar. |
| Approximate arithmetic hides errors at small n | Approximation width is a parameter. Use exact mode for all toy-curve tests, and count errors with a sampling harness at large n. |
| Compile times blow up at large n | `callback_threshold`, `qache` on per-window kernels, and resource counts from single-window traces (`total = windows × one window + overhead`). |

## 7. Decision log: options considered and set aside

- **Projective or Jacobian coordinates**, including the x-only Montgomery ladder proposed in PR #15. Avoiding per-addition inversion is attractive, but the projective representation is not unique. In Shor's algorithm the accumulator must hold a canonical value, or interference is destroyed. Making it canonical needs a division per addition anyway: Häner et al. 2020, App. A.4, count it as 2 additions + 2 divisions, worse than affine. CFS 2026 avoid this only by compressing the output to a Legendre-symbol bit at Õ(n⁴) gate cost. The x-only ladder adds further garbage (the second ladder register and Z) that must be uncomputed. PR #15 remains useful as a classical reference and as a source of tests, not as the target design.
- **Bernstein–Yang safegcd inversion.** Häner et al. (App. A.3) find that the recursive form needs quantum matrix products with no in-place method, and that its base case costs about one Kaliski round. No prime-field quantum design uses it.
- **Keeping the 2n-qubit exponent registers.** These dominate qubits for no benefit once the semiclassical QFT is available in jasp.

## 8. References

The headline figures for the starred items were re-checked against their arXiv abstracts in November 2026. The others are summarised from abstracts, and some unresolved discrepancies are flagged above.

- ★ R. Babbush et al., *Securing Elliptic Curve Cryptocurrencies against Quantum Vulnerabilities: Resource Estimates and Mitigations*, [arXiv:2603.28846](https://arxiv.org/abs/2603.28846) (2026).
- ★ A. Schrottenloher, *Optimized Point Addition Circuits for Elliptic Curve Discrete Logarithms*, [arXiv:2606.02235](https://arxiv.org/abs/2606.02235) / ePrint 2026/1128 (2026).
- ★ D. Litinski, *How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates*, [arXiv:2306.08585](https://arxiv.org/abs/2306.08585) (2023).
- D. Polimeni, R. Seidel, *End-to-end compilable implementation of quantum elliptic curve logarithm in Qrisp*, [arXiv:2501.10228](https://arxiv.org/abs/2501.10228) (2025).
- M. Roetteler, M. Naehrig, K. Svore, K. Lauter, [arXiv:1706.06752](https://arxiv.org/abs/1706.06752) (2017).
- T. Häner, S. Jaques, M. Naehrig, M. Roetteler, M. Soeken, [arXiv:2001.09580](https://arxiv.org/abs/2001.09580) (2020).
- C. Gidney, *Windowed quantum arithmetic*, [arXiv:1905.07682](https://arxiv.org/abs/1905.07682); C. Gidney, M. Ekerå, [arXiv:1905.09749](https://arxiv.org/abs/1905.09749) (2019).
- R. Babbush et al., unary-iteration QROM, [arXiv:1805.03662](https://arxiv.org/abs/1805.03662) (2018); G. H. Low, V. Kliuchnikov, L. Schaeffer, [arXiv:1812.00954](https://arxiv.org/abs/1812.00954); D. Berry et al., [arXiv:1902.02134](https://arxiv.org/abs/1902.02134); T. Khattar, C. Gidney, [arXiv:2407.17966](https://arxiv.org/abs/2407.17966).
- M. Ekerå, *Revisiting Shor's quantum algorithm for computing general discrete logarithms*, [arXiv:1905.09084](https://arxiv.org/abs/1905.09084); M. Ekerå, J. Håstad, [arXiv:1702.00249](https://arxiv.org/abs/1702.00249).
- É. Gouzien et al., cat-qubit ECDLP-256, [arXiv:2302.06639](https://arxiv.org/abs/2302.06639) (2023).
- G. Banegas, D. J. Bernstein, I. van Hoof, T. Lange, binary curves, ePrint 2020/1296 (TCHES 2021).
- Garn, Kan, windowed binary-curve design, [arXiv:2503.02984](https://arxiv.org/abs/2503.02984) (2025).
- Khattar et al., [arXiv:2510.10967](https://arxiv.org/abs/2510.10967) (Euclid dialog / Bézout reconstruction).
- Luo et al., [arXiv:2607.13816](https://arxiv.org/abs/2607.13816) (2026); Chevignard, Fouque, Schrottenloher, ePrint 2026/280 (2026); Häner et al. (IonQ), ePrint 2026/1916 (2026); ECDSA.Fail, [arXiv:2609.09582](https://arxiv.org/abs/2609.09582) (2026).
- C. Gidney, *How to factor 2048 bit RSA integers with less than a million noisy qubits*, [arXiv:2505.15917](https://arxiv.org/abs/2505.15917) (2025).
- Test instances: "Brace for impact", [arXiv:2508.14011](https://arxiv.org/abs/2508.14011) (secp256k1-shaped curves from 6 to 256 bits).
