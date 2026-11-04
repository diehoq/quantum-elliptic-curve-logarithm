# Inversion-free x-only Montgomery ladder (alternative point arithmetic)

## Summary

This contribution adds an **inversion-free x-only (projective `X:Z`) Montgomery
ladder** as an alternative to the affine double-and-add engine in
`src/quantum/ec_arithmetic.py`.

The existing `q_ec_add_inpl` computes an affine slope, so it runs
`kaliski_mod_inv` **twice per point addition** (compute λ + uncompute λ). Modular
inversion is the single most expensive reversible primitive (Kaliski = `2n` loop
iterations), so an `n`-bit scalar costs on the order of **~4n Kaliski inversions**
on the critical path.

The Montgomery ladder in projective `(X:Z)` coordinates removes every
intermediate inversion: each step is built only from field
multiply/square/add (`xDBL`, `xADD`), and a **single** inversion at the very end
deaffinifies `x = X/Z`.

## Files

| File | Purpose |
|------|---------|
| `src/classical/xonly_reference.py` | Classical x-only ladder + field-op counter (ground-truth + resource model) |
| `src/quantum/ec_arithmetic_xonly.py` | Qrisp `q_xDBL`, `q_xADD`, classical-`k` ladder, and quantum-`k` (controlled-swap) ladder |
| `tests/test_xonly_reference.py` | Validates the x-only ladder against this repo's affine group law on **all** `curves_and_keys.json` curves; checks the inversion saving |
| `tests/test_xonly_quantum.py` | Validates the Qrisp primitives + ladders under `boolean_simulation` on `p=13` |

## Results

**Classical (all 17 QDay curves, `a=0, b=7`)** — the x-only ladder reproduces
every public key's x-coordinate and the loop uses exactly **one** inversion
regardless of bit size, versus the `bit_length+hamming_weight-2` inversions of
affine double-and-add:

```
inversions in the scalar-mult loop:   x-only = 1   |   affine ≈ 2n  (→ ~4n Kaliski in q_ec_add_inpl)
17/17 curves: ladder_x(k, x(G)) == x(public_key)
```

**Quantum (`p=13`, `y²=x³+7`, `G=(11,5)`, `boolean_simulation`)**

- `q_xDBL(G)` → `x(2G) = 7` ✓
- `q_xADD(2G, G; diff=G)` → `x(3G) = 8` ✓
- classical-`k` ladder: `x([2]G)=7`, `x([3]G)=8`, `x([5]G)=7` ✓
- **quantum-`k` ladder (real Shor form)**: the scalar bits drive
  controlled-swaps of `(R0,R1)`; `k=2` (cswap inactive) → `7`, `k=3`
  (cswap active) → `8` ✓

## Qrisp version note (important)

Validated on **qrisp 0.9.5** (PyPI). `QuantumModulus`'s montgomery-shift
behaviour differs between releases; three rules (documented inline in
`ec_arithmetic_xonly.py`) make multiply/add/sub compose under
`boolean_simulation`:

1. Plain `A*B` decodes correctly and chains; a classical `*int` **resets** the
   montgomery shift to 0 (the normalizer).
2. `+`/`-` need equal shift: combine same-depth operands directly; when depths
   differ, normalize **exactly one** side (`*1`). `_std(A) - _std(B)` is wrong.
3. Never square a register by itself (`C*C`): use two independent products,
   e.g. `(X1*X2)*(X1*X2)`.

On the repo's pinned `0.8.2` branch, `QuantumModulus` has a montgomery
`boolean_simulation` uncomputation bug (the repo's own `boolean_sim` test shows
it), so these primitives should run against **qrisp ≥ 0.9.5**.

## Limitations / future work

- `boolean_simulation` takes classical inputs, so the quantum tests validate
  **basis states** of `k` (the controlled-swap circuit is identical in
  superposition — the genuine Shor use). True superposition needs statevector
  simulation, which is qubit-limited.
- qrisp 0.9.5's XLA compile is slow for deep circuits; the quantum-`k` ladder is
  validated at `n=2` (one cswap iteration). Larger `n`/`p` need a faster
  compiler or more memory.
- The double-scalar `[k]G + [l]Q` Shor oracle is two ladders + one affine add;
  composing it end-to-end in Qrisp is the natural next step.

## How to run

```bash
pip install "qrisp>=0.9.5" pytest
pytest tests/test_xonly_reference.py -q      # fast, no qrisp needed
pytest tests/test_xonly_quantum.py -q        # slow (boolean_simulation JIT)
```
