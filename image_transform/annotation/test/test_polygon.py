import numpy as np
import pytest

from ..polygon import Polygon
from ..utils import IsWithinImage, HALF_PI


_ERROR_TOL = 1e-5
img_w, img_h = 100, 100


def is_close(f1: float, f2: float):
    return abs(f1 - f2) <= _ERROR_TOL


def test_valid():
    with pytest.raises(AssertionError):
        _ = Polygon([(0, 0), (10, 10), (0, 10), (10, 0)], img_w, img_h)

    with pytest.raises(AssertionError):
        _ = Polygon([(0, 0), (10, 10)], img_w, img_h)


def test_area():
    P = Polygon([(0, 0), (0, 10), (10, 0)], img_w, img_h)
    assert is_close(P.signed_area, -50)
    P = Polygon([(0, 0), (10, 0), (0, 10)], img_w, img_h)
    assert is_close(P.signed_area, 50)


def test_intersection_area():
    P = Polygon([(0, 0), (0, 20), (20, 20), (20, 0)], img_w, img_h)
    Q = Polygon([(10, 10), (10, 30), (30, 30), (30, 10)], img_w, img_h)
    assert is_close(P.intersection_area(Q), 100)
    assert is_close(P.union_area(Q), 700)
    assert is_close(P.iou(Q), 100/700)

    P = Polygon([(0, 10), (0, 20), (30, 20), (30, 10)], img_w, img_h)
    Q = Polygon([(10, 0), (10, 30), (20, 30), (20, 0)], img_w, img_h)
    assert is_close(P.intersection_area(Q), 100)

    P = Polygon([(0, 0), (0, 50), (20, 50), (20, 0)], img_w, img_h)
    Q = Polygon([
        (10, 10), (10, 20), (30, 20), (30, 30),
        (10, 30), (10, 40), (40, 40), (40, 10)
    ], img_w, img_h)
    assert is_close(P.intersection_area(Q), 200)

    P = Polygon([(10, 10), (10, 40), (40, 40), (40, 10)], img_w, img_h)
    Q = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)], img_w, img_h)
    assert is_close(P.intersection_area(Q), 100)

    P = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)], img_w, img_h)
    Q = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)], img_w, img_h)
    assert is_close(P.intersection_area(Q), 0)

    P = Polygon([(0, 0), (0, 10), (10, 0)], img_w, img_h)
    area = P.intersection_area(P)
    assert is_close(area, P.area)

    P = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)], img_w, img_h)
    area = P.intersection_area(P)
    assert is_close(area, P.area)

    P = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)], img_w, img_h)
    Q = Polygon([(10, 10), (10, 20), (20, 20), (20, 10)], img_w, img_h)
    assert is_close(P.intersection_area(Q), 0)

    P = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)], img_w, img_h)
    Q = Polygon([(10, 20), (20, 20), (20, 10), (10, 10)], img_w, img_h)
    assert is_close(P.intersection_area(Q), 0)


def test_transform():
    p = Polygon([(10, 10), (10, 60), (180, 10)], 100, 100)
    assert p.check_within_image() == IsWithinImage.PARTIAL
    with pytest.raises(NotImplementedError):
        p.clip()

    p = Polygon([(-10, -10), (-10, -60), (-180, -10)], 100, 100)
    assert p.check_within_image() == IsWithinImage.NO

    p = Polygon([(10, 10), (10, 60), (80, 10)], 100, 100)
    assert p.check_within_image() == IsWithinImage.YES
    mask = p.to_mask()
    assert len(mask.scope) == 1 and np.sum(mask.mask > 0) == 1846
    array = p.to_numpy()
    assert np.all(array == np.array([(10, 10), (10, 60), (80, 10)]))
    assert p, Polygon.from_numpy(array, 100, 100)
    _ = str(p)
    assert p == p.clip()
    assert p.pad(10, 20, 30, 40) == Polygon([(40, 20), (40, 70), (110, 20)], 170, 130)
    assert p.crop(10, 94, 10, 94) == Polygon([(0, 0), (0, 50), (70, 0)], 85, 85)
    assert p == p.horizontal_flip().horizontal_flip()
    assert p == p.vertical_flip().vertical_flip()
    _ = p.rotate(HALF_PI/2)
    _ = p._rotate_right_angle(90)
    assert p.resize(200, 200) == Polygon([(20, 20), (20, 120), (160, 20)], 200, 200)
    assert p.rescale(2) == Polygon([(20, 20), (20, 120), (160, 20)], 200, 200)
    assert p.transpose() == Polygon([(10, 10), (60, 10), (10, 80)], 100, 100)
