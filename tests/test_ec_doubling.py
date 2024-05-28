import src.classical.ec_arithmetic as clECarithm
import src.quantum.ec_arithmetic as qECarithm

def test_ec_doubling():
    curve = clECarithm.EllCurve(a=5, b=4, p=7)
    input_points = [(0,2),(0,5),(2,1),(2,6),(3,2),(3,5),(4,2),(4,5)] 

    for point in input_points:
        point_ell = clECarithm.EllPoint(point[0], point[1])

        result_ell = clECarithm.ell_double(point_ell, curve)
        result_qrisp = qECarithm.qrisp_ell_double(point, curve)

        assert result_ell.x == result_qrisp[0]
        assert result_ell.y == result_qrisp[1]
