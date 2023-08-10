import numpy as np
import pytest

from ..image import EmptyImage
from ..point import Point
from ..point_array import PointArray
from ..box import Box
from ..oriented_box import OrientedBox
from ..box_array import BoxArray
from ..oriented_box_array import OrientedBoxArray
from ..utils import IsWithinImage, HALF_PI, ONE_AND_HALF_PI
from ..label import Scoped, ArbitraryHashable, Empty


def test_box():
    box = Box(50, 55, 10, 20, 40, 40)
    assert box.check_within_image() == IsWithinImage.NO
    box = Box(0, 10, 0, 20, 5, 5)
    assert box.check_within_image() == IsWithinImage.PARTIAL
    assert box.clip() == Box(0, 4, 0, 4, 5, 5)

    box = Box(0, 10, 0, 20, 40, 40)
    assert box.check_within_image() == IsWithinImage.YES
    assert box.img_w == 40 and box.img_h == 40
    assert np.allclose(box.aspect_ratio, 0.5)
    _ = str(box)
    array = box.to_numpy()
    assert np.all(array == np.array([0, 10, 0, 20]))
    assert box == Box.from_numpy(array, 40, 40)
    tl, tr, bl, br = box.to_points()
    assert tl == Point(0, 0, 40, 40)
    assert tr == Point(10, 0, 40, 40)
    assert bl == Point(0, 20, 40, 40)
    assert br == Point(10, 20, 40, 40)
    tl, tr, br, bl = box.to_polygon().points
    assert tl == Point(0, 0, 40, 40)
    assert tr == Point(10, 0, 40, 40)
    assert bl == Point(0, 20, 40, 40)
    assert br == Point(10, 20, 40, 40)
    new_box = Box.from_corner_points(tl, br, 40, 40)
    assert new_box == box
    cx, cy, w, h = box.to_numpy_as_cxcywh()
    assert cx == 5 and cy == 10 and w == 10 and h == 20
    new_box = Box.from_numpy_as_cxcywh(np.array([cx, cy, w, h]), 40, 40)
    assert new_box == box
    xmin, ymin, w, h = box.to_numpy_as_xywh()
    assert xmin == 0 and ymin == 0 and w == 10 and h == 20
    new_box = Box.from_numpy_as_xywh(np.array([xmin, ymin, w, h]), 40, 40)
    assert new_box == box
    assert box.clip() == box
    assert box == box.pad(up=1, down=2, left=3, right=4).crop(xmin=3, xmax=42, ymin=1, ymax=40)
    assert box == box.vertical_flip().vertical_flip()
    assert box == box.horizontal_flip().horizontal_flip()
    assert box.rotate(HALF_PI/2) == \
        OrientedBox(5, 10, 10, 20, ONE_AND_HALF_PI, 40, 40).rotate(HALF_PI/2)
    assert box.rotate_right_angle(90) == Box(20, 40, 0, 10, 40, 40)
    assert box.rotate_right_angle(180) == Box(30, 40, 20, 40, 40, 40)
    assert box.rotate_right_angle(270) == Box(0, 20, 30, 40, 40, 40)
    assert box.rotate_right_angle(360) == box
    assert box.transpose() == Box(0, 20, 0, 10, 40, 40)

    box_other = Box(0, 20, 0, 10, 40, 40)
    assert np.allclose(box.intersection_area(box_other), 100)
    assert np.allclose(box.union_area(box_other), 300)
    assert np.allclose(box.iou(box_other), 100/300)

    box_other = Box(-10, -1, -10, -1, 40, 40)
    assert np.allclose(box.intersection_area(box_other), 0)
    assert np.allclose(box.iou(box_other), 0)


def test_box_array():
    assert BoxArray(
        xmin=[-10, -10], xmax=[-5, -2], ymin=[0, 0], ymax=[20, 10], img_w=40, img_h=40
    ).check_within_image() == IsWithinImage.NO
    assert BoxArray(
        xmin=[-10, -5], xmax=[-5, 5], ymin=[0, 0], ymax=[20, 10], img_w=40, img_h=40
    ).check_within_image() == IsWithinImage.PARTIAL
    assert BoxArray(
        xmin=[-10, -5], xmax=[-5, 5], ymin=[0, 0], ymax=[20, 10], img_w=40, img_h=40
    ).clip().check_within_image() == IsWithinImage.YES
    assert BoxArray(
        xmin=[-10, -5], xmax=[-5, 5], ymin=[0, 0], ymax=[20, 10], img_w=40, img_h=40
    ).clip().clip().check_within_image() == IsWithinImage.YES
    boxarr = BoxArray(
        xmin=[0, 0], xmax=[10, 20], ymin=[0, 0], ymax=[20, 10], img_w=40, img_h=40
    )
    assert boxarr.check_within_image() == IsWithinImage.YES
    assert boxarr.select([1, 0]) == BoxArray(
        xmin=[0, 0], xmax=[20, 10], ymin=[0, 0], ymax=[10, 20], img_w=40, img_h=40
    )
    assert np.allclose(boxarr.aspect_ratio, np.array([0.5, 2.]))
    with pytest.raises(IndexError):
        _ = boxarr[2]

    assert isinstance(repr(boxarr), str)

    tl, tr, bl, br = boxarr.to_points()
    assert tl == PointArray([0, 0], [0, 0], 40, 40)
    assert tr == PointArray([10, 20], [0, 0], 40, 40)
    assert bl == PointArray([0, 0], [20, 10], 40, 40)
    assert br == PointArray([10, 20], [20, 10], 40, 40)
    assert BoxArray.from_corner_points(tl, br) == boxarr

    assert boxarr == BoxArray.from_boxes(boxarr.to_boxes())

    boxarr = BoxArray([0], [10], [0], [20], 40, 40)
    assert boxarr.to_oriented_box_array() == \
        OrientedBoxArray([5], [10], [10], [20], [ONE_AND_HALF_PI], 40, 40)
    assert boxarr.rotate_right_angle(90) == BoxArray([20], [40], [0], [10], 40, 40)
    assert boxarr.rotate_right_angle(180) == BoxArray([30], [40], [20], [40], 40, 40)
    assert boxarr.rotate_right_angle(270) == BoxArray([0], [20], [30], [40], 40, 40)
    assert boxarr.rotate_right_angle(360) == boxarr
    assert boxarr.transpose() == BoxArray([0], [20], [0], [10], 40, 40)

    boxarr = BoxArray(
        xmin=[0, 0], xmax=[10, 20], ymin=[0, 0], ymax=[20, 10], img_w=40, img_h=40
    )
    other = boxarr.transpose()
    assert np.allclose(boxarr.intersection_area(other), np.array([[100, 200], [200, 100]]))
    assert np.allclose(boxarr.union_area(other), np.array([[300, 200], [200, 300]]))
    assert np.allclose(boxarr.iou(other), np.array([[1/3, 1], [1, 1/3]], float))

    assert boxarr == BoxArray.from_numpy(boxarr.to_numpy(), 40, 40)
    assert boxarr == BoxArray.from_numpy_as_cxcywh(boxarr.to_numpy_as_cxcywh(), 40, 40)
    assert boxarr == BoxArray.from_numpy_as_xywh(boxarr.to_numpy_as_xywh(), 40, 40)
    assert boxarr == BoxArray.from_numpy_as_xyxy(boxarr.to_numpy_as_xyxy(), 40, 40)

    new_boxarr = boxarr + boxarr[0]
    assert new_boxarr == BoxArray(
        xmin=[0, 0, 0], xmax=[10, 20, 10], ymin=[0, 0, 0], ymax=[20, 10, 20], img_w=40, img_h=40
    )
    new_boxarr = boxarr + boxarr
    assert new_boxarr == BoxArray(
        xmin=[0, 0, 0, 0], xmax=[10, 20, 10, 20], ymin=[0, 0, 0, 0], ymax=[20, 10, 20, 10], img_w=40, img_h=40
    )
    with pytest.raises(TypeError):
        _ = boxarr + EmptyImage(40, 40)

    boxarr = BoxArray(
        xmin=[0, 0, 0, 0, 0],
        xmax=[1, 1, 5, 6, 1],
        ymin=[0, 0, 0, 0, 0],
        ymax=[3, 2, 5, 6, 1],
        img_w=10,
        img_h=10
    )
    boxarr_nms = boxarr.nms(0.5)
    assert len(boxarr_nms) == 3
    assert len(boxarr_nms.label) == 3 and all(isinstance(label, Empty) for label in boxarr_nms.label)

    with pytest.raises(AssertionError):
        boxarr = BoxArray(
            xmin=[0, 0],
            xmax=[1, 1],
            ymin=[0, 0],
            ymax=[3, 2],
            img_w=10,
            img_h=10,
            label=[ArbitraryHashable(None)]
        )

    class MyScoped(Scoped):
        scope = ["a", "b"]

    with pytest.raises(AssertionError):
        boxarr = BoxArray(
            xmin=[0, 0],
            xmax=[1, 1],
            ymin=[0, 0],
            ymax=[3, 2],
            img_w=10,
            img_h=10,
            label=[ArbitraryHashable(None), MyScoped("a")]
        )


def test_empty_box_array():
    a = BoxArray.from_boxes([], 50, 50)

    b = a.nms()
    b.check()
    assert len(b) == 0
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

    m = a.intersection_area(a)
    assert m.shape == (0, 0)
    m = a.union_area(a)
    assert m.shape == (0, 0)
    assert a == a
    assert len(a + a) == 0
