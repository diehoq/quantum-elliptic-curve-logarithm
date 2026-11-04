# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report them privately through GitHub's
[private vulnerability reporting](https://github.com/diehoq/quantum-elliptic-curve-logarithm/security/advisories/new)
(the **Security → Report a vulnerability** tab of this repository).

Include the affected file or dependency, steps to reproduce, and the impact you expect.
You should get a reply within a few days.

## Scope

This is a research codebase: quantum circuits for elliptic-curve arithmetic, run locally
on Qrisp's simulator. It does not run a service or handle credentials. In scope:

- the code in `src/`, `tests/` and the notebooks
- the CI workflow in `.github/workflows/`
- the pinned dependencies in `requirements.txt`

The keys in `curves_and_keys.json` are the public QDay Prize challenge instances. They
are not secrets.
