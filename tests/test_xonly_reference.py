"""Classical validation of the inversion-free x-only ladder against this repo's
own affine group law, on every curve in curves_and_keys.json, plus the
inversion-count comparison that motivates the contribution."""
import json
import os

import pytest

import src.classical.ec_arithmetic as cl
import src.classical.xonly_reference as xo

_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "..", "curves_and_keys.json")) as _fh:
    CURVES = json.load(_fh)

A, B = 0, 7  # QDay / secp256k1 family


@pytest.mark.parametrize("c", CURVES, ids=lambda c: f"{c['bit_length']}bit")
def test_xonly_ladder_matches_affine_reference(c):
    p = c["prime"]
    G = tuple(c["generator_point"])
    k = c["private_key"]
    pub = tuple(c["public_key"])

    # sanity: the curve really is y^2 = x^3 + 7
    assert (G[1] * G[1] - G[0] ** 3 - B) % p == 0

    # ground truth from this repo's own classical group law
    curve = cl.EllCurve(a=A, b=B, p=p)
    ref = cl.ell_mult_add(cl.EllPoint(*G), cl.EllZero, k, curve)
    assert (ref.x, ref.y) == pub

    # x-only ladder must reproduce the public key's x-coordinate
    F = xo.FieldCounter(p)
    x = xo.ladder_x(k, G[0], A, B, F)
    assert x == pub[0]

    # ...using exactly ONE modular inversion (the final X/Z deaffinify),
    # whereas affine double-and-add needs one inversion per doubling/addition
    # (>= bit_length-1), each of which becomes TWO Kaliski inversions in the
    # current quantum q_ec_add_inpl.
    assert F.inv == 1
    affine_inv = xo.affine_doubleadd_inversions(k)
    assert affine_inv >= k.bit_length() - 1
    assert affine_inv > F.inv


def test_inversion_saving_is_independent_of_bitsize():
    """The x-only loop is inversion-free at every size; affine grows ~2n."""
    for c in CURVES:
        F = xo.FieldCounter(c["prime"])
        xo.ladder_x(c["private_key"], c["generator_point"][0], A, B, F)
        assert F.inv == 1
