import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm
import qrisp
import pytest

curves_with_points = [
    # {
    #     "curve_params": {"a": 0, "b": 3, "p": 5},
    #     "points": [(2, 1), (3, 0), (2, 4)],
    #     "base_point": (1, 2),
    # },
    # NOTICE: A CHAIN SHOULD BE CHOSEN FOR EACH CURVE BEFORE LAUNCHING THE TESTS
    {
        "curve_params": {"a": 5, "b": 4, "p": 7},
        "points": [
            (2, 1),
            (4, 5),
            (0, 2),
            (5, 0),
            (0, 5),
            (4, 2),
        ],
        "base_point": (3, 5),
    },
    # {
    #     "curve_params": {"a": 1, "b": 6, "p": 11},
    #     "points": [(2, 4), (2, 7), (5, 2), (7, 2), (7, 9), (8, 3), (10, 2)],
    #     "base_point": (5, 2),
    # },
    # {
    #     "curve_params": {"a": 2, "b": 3, "p": 13},
    #     "points": [
    #         (0, 4),
    #         (0, 9),
    #         (3, 6),
    #         (3, 7),
    #         (4, 6),
    #         (4, 7),
    #         (6, 6),
    #         (6, 7),
    #         (7, 3),
    #         (7, 10),
    #         (9, 3),
    #         (9, 10),
    #         (10, 3),
    #         (10, 10),
    #         (11, 2),
    #         (11, 11),
    #     ],
    #     "base_point": (3, 6),
    # },
    # {
    #     "curve_params": {"a": 1, "b": 5, "p": 17},
    #     "points": [
    #         (2, 7),
    #         (2, 10),
    #         (3, 1),
    #         (3, 16),
    #         (5, 4),
    #         (5, 13),
    #         (7, 7),
    #         (7, 10),
    #         (8, 7),
    #         (8, 10),
    #         (11, 2),
    #         (11, 15),
    #         (14, 3),
    #         (14, 14),
    #     ],
    #     "base_point": (5, 4),
    # },
    # {
    #     "curve_params": {"a": 2, "b": 8, "p": 19},
    #     "points": [
    #         (1, 7),
    #         (1, 12),
    #         (2, 1),
    #         (2, 18),
    #         (4, 2),
    #         (4, 17),
    #         (7, 2),
    #         (7, 17),
    #         (8, 2),
    #         (8, 17),
    #         (14, 5),
    #         (14, 14),
    #         (18, 9),
    #         (18, 10),
    #     ],
    #     "base_point": (1, 7),
    # },
]


@pytest.mark.parametrize("curve_data", curves_with_points)
def test_ec_addition(curve_data):

    curve_params = curve_data["curve_params"]
    points = curve_data["points"]
    base_point_coords = curve_data["base_point"]

    curve = clECarithm.EllCurve(
        a=curve_params["a"], b=curve_params["b"], p=curve_params["p"]
    )

    base_point_ell = clECarithm.EllPoint(base_point_coords[0], base_point_coords[1])
    base_point_qrisp = list(base_point_coords)

    for point in points:
        P_ell = clECarithm.EllPoint(point[0], point[1])
        anc_0 = qrisp.QuantumModulus(curve_params["p"])
        anc_0[:] = point[0]
        anc_1 = qrisp.QuantumModulus(curve_params["p"])
        anc_1[:] = point[1]

        result_ell = clECarithm.ell_add_generic(P_ell, base_point_ell, curve)
        qECarithm.qrisp_ell_add_inpl(
            [anc_0, anc_1], base_point_qrisp, curve_params["p"]
        )

        assert qrisp.multi_measurement([anc_0, anc_1]) == {
            (result_ell.x, result_ell.y): 1
        }, f"Coordinate mismatch for point {point} on curve with a={curve.a}, b={curve.b}, p={curve.p}"
