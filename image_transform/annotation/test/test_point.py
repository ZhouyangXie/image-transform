import numpy as np
import pytest

from ..label import Empty
from ..point import Point, line_intersection
from ..point_array import PointArray
from ..utils import IsWithinImage, HALF_PI


def test_point():
    point = Point(5, -5, 10, 10)
    assert point.check_within_image() == IsWithinImage.NO
    assert point == point.copy()
    point = Point(2, 3, 10, 10, 'a point')
    assert point.check_within_image() == IsWithinImage.YES
    assert point.img_w == 10 and point.img_h == 10
    array = point.to_numpy()
    _ = str(point)
    assert np.all(array == np.array([2, 3], int))
    assert point == Point.from_numpy(array, 10, 10, 'a point')
    assert point == point.clip()
    assert point == point.pad(1, 2, 3, 4).crop(3, 12, 1, 10)
    assert point == point.vertical_flip().vertical_flip()
    assert point == point.horizontal_flip().horizontal_flip()
    assert point.rotate(HALF_PI/2) == Point(
        9/np.sqrt(2), 5/np.sqrt(2), 10 * np.sqrt(2), 10 * np.sqrt(2), 'a point')
    assert point.rotate_right_angle(90) == Point(7, 2, 10, 10, 'a point')
    assert point.resize(20, 20) == Point(4, 6, 20, 20, 'a point')
    assert point.rescale(3, 2) == Point(6, 6, 30, 20, 'a point')
    assert point.transpose() == Point(3, 2, 10, 10, 'a point')


def test_point_array():
    pa = PointArray([5], [-5], 10, 10)
    assert pa.check_within_image() == IsWithinImage.NO
    pa = PointArray([2, 5], [3, -5], 10, 10)
    assert pa.check_within_image() == IsWithinImage.PARTIAL
    pa = pa.clip()
    assert len(pa.clip()) == 1
    assert pa.img_w == 10 and pa.img_h == 10
    with pytest.raises(TypeError):
        pa = PointArray([2, 5], [3, -5], 10, 10, [np.zeros(1), None])
    pa = PointArray([2], [3], 10, 10, [None])
    assert all(isinstance(_label, Empty) for _label in pa.label)

    array = pa.to_numpy()
    assert np.all(array == np.array([[2, 3]], int))
    assert pa == PointArray.from_numpy(array, 10, 10)
    assert pa == pa.pad(1, 2, 3, 4).crop(3, 12, 1, 10)
    assert pa == pa.vertical_flip().vertical_flip()
    assert pa == pa.horizontal_flip().horizontal_flip()
    assert pa.rotate(HALF_PI/2) == PointArray(
        [9/np.sqrt(2)], [5/np.sqrt(2)], 10 * np.sqrt(2), 10 * np.sqrt(2))
    assert pa.rotate_right_angle(90) == PointArray([7], [2], 10, 10)
    assert pa.resize(20, 20) == PointArray([4], [6], 20, 20)
    assert pa.rescale(3, 2) == PointArray([6], [6], 30, 20)
    assert pa.transpose() == PointArray([3], [2], 10, 10)

    pa = PointArray([2, 5], [3, -5], 10, 10)


def test_empty_point_array():
    a = PointArray.from_points([], 40, 40)
    b = a.clip()
    b.check()
    assert len(b) == 0
    b = a.pad(1, 2, 3, 4)
    b.check()
    assert len(b) == 0
    b = a.crop(1, 2, 3, 4)
    b.check()
    assert len(b) == 0
    b = a.horizontal_flip()
    b.check()
    assert len(b) == 0
    b = a.vertical_flip()
    b.check()
    assert len(b) == 0
    b = a.rotate(10)
    b.check()
    assert len(b) == 0
    b = a.rotate_right_angle(270)
    b.check()
    assert len(b) == 0
    b = a.resize(20, 20)
    b.check()
    assert len(b) == 0
    b = a.transpose()
    b.check()
    assert len(b) == 0

    assert a == a
    assert len(a + a) == 0


def test_line_intersection():
    img_w, img_h = 100, 100
    inter = line_intersection(
        (Point(0, 0, img_w, img_h), Point(10, 10, img_w, img_h)),
        (Point(0, 10, img_w, img_h), Point(10, 0, img_w, img_h))
    )
    assert inter == Point(5, 5, img_w, img_h)
    inter = line_intersection(
        (Point(0, 0, img_w, img_h), Point(10, 10, img_w, img_h)),
        (Point(2, 2, img_w, img_h), Point(5, 5, img_w, img_h))
    )
    assert inter is None
    inter = line_intersection(
        (Point(0, 0, img_w, img_h), Point(10, 10, img_w, img_h)),
        (Point(20, 0, img_w, img_h), Point(0, 20, img_w, img_h))
    )
    assert inter is None
    inter = line_intersection(
        (Point(10, 10, img_w, img_h), Point(0, 0, img_w, img_h)),
        (Point(20, 0, img_w, img_h), Point(0, 20, img_w, img_h))
    )
    assert inter == Point(10, 10, img_w, img_h)
    inter = line_intersection(
        (Point(0, 0, img_w, img_h), Point(10, 10, img_w, img_h)),
        (Point(10, 10, img_w, img_h), Point(20, 20, img_w, img_h))
    )
    assert inter is None
    inter = line_intersection(
        (Point(10, 10, img_w, img_h), Point(0, 0, img_w, img_h)),
        (Point(10, 10, img_w, img_h), Point(20, 20, img_w, img_h))
    )
    assert inter == Point(10, 10, img_w, img_h)
    inter = line_intersection(
        (Point(0, 10, img_w, img_h), Point(0, 0, img_w, img_h)),
        (Point(0, 10, img_w, img_h), Point(10, 10, img_w, img_h))
    )
    assert inter == Point(0, 10, img_w, img_h)
    inter = line_intersection(
        (Point(5, 0, img_w, img_h), Point(5, 10, img_w, img_h)),
        (Point(0, 5, img_w, img_h), Point(10, 5, img_w, img_h))
    )
    assert inter == Point(5, 5, img_w, img_h)
