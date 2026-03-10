import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm
import pytest

curves_with_points = [
    {
        "curve_params": {"a": 0, "b": 3, "p": 5},
        "points": [(1, 2), (1, 3), (2, 1), (2, 4)],
    },
    {
        "curve_params": {"a": 5, "b": 4, "p": 7},
        "points": [(0, 2), (0, 5), (2, 1), (2, 6), (3, 2), (3, 5), (4, 2), (4, 5)],
    },
    {
        "curve_params": {"a": 1, "b": 6, "p": 11},
        "points": [(2, 4), (2, 7), (5, 2), (7, 2), (7, 9), (8, 3), (10, 2)],
    },
    {
        "curve_params": {"a": 2, "b": 3, "p": 13},
        "points": [
            (0, 4),
            (0, 9),
            (3, 6),
            (3, 7),
            (4, 6),
            (4, 7),
            (6, 6),
            (6, 7),
            (7, 3),
            (7, 10),
            (9, 3),
            (9, 10),
            (10, 3),
            (10, 10),
            (11, 2),
            (11, 11),
        ],
    },
    {
        "curve_params": {"a": 1, "b": 5, "p": 17},
        "points": [
            (2, 7),
            (2, 10),
            (3, 1),
            (3, 16),
            (5, 4),
            (5, 13),
            (7, 7),
            (7, 10),
            (8, 7),
            (8, 10),
            (11, 2),
            (11, 15),
            (14, 3),
            (14, 14),
        ],
    },
    {
        "curve_params": {"a": 2, "b": 8, "p": 19},
        "points": [
            (1, 7),
            (1, 12),
            (2, 1),
            (2, 18),
            (4, 2),
            (4, 17),
            (7, 2),
            (7, 17),
            (8, 2),
            (8, 17),
            (14, 5),
            (14, 14),
            (18, 9),
            (18, 10),
        ],
    },
]


@pytest.mark.parametrize("curve_data", curves_with_points)
def test_ec_doubling(curve_data):

    curve_params = curve_data["curve_params"]
    points = curve_data["points"]

    curve = clECarithm.EllCurve(
        a=curve_params["a"], b=curve_params["b"], p=curve_params["p"]
    )

    for point in points:
        point_ell = clECarithm.EllPoint(point[0], point[1])

        result_ell = clECarithm.ell_double(point_ell, curve)
        result_qrisp = qECarithm.q_ec_double(point, curve)

        assert result_ell.x == result_qrisp[0]
        assert result_ell.y == result_qrisp[1]
