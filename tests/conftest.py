import contextlib
import sys

import jax
import pytest

FAULTY_MARKER = "Faulty uncomputation"


@pytest.fixture
def no_faulty_uncomputation(capfd):
    """Context manager that fails the test if Qrisp reports faulty uncomputation.

    Qrisp's jasp simulators report it through ``jax.debug.print`` (stdout), not the
    ``warnings`` module, so it has to be read from the captured output.

    Usage::

        with no_faulty_uncomputation("point (0, 2)"):
            run_circuit()
    """

    @contextlib.contextmanager
    def guard(context=""):
        capfd.readouterr()  # drop output produced before the guarded call
        yield
        jax.effects_barrier()  # make sure debug prints have been flushed
        out, err = capfd.readouterr()
        # Re-emit so the output still shows up in pytest's report.
        sys.stdout.write(out)
        sys.stderr.write(err)
        where = f" for {context}" if context else ""
        assert FAULTY_MARKER not in out + err, f"Qrisp reported faulty uncomputation{where}"

    return guard
