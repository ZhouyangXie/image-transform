from ..oriented_box import OrientedBox
from ..oriented_box_array import OrientedBoxArray
from ..utils import IsWithinImage, HALF_PI


def test_obox_simple():
    ob = OrientedBox(0, 0, 3, 2, HALF_PI/3, 40, 40)
    assert ob.check_within_image() == IsWithinImage.PARTIAL
    ob = OrientedBox(-5, -5, 3, 2, HALF_PI/3, 40, 40)
    assert ob.check_within_image() == IsWithinImage.NO
    ob = OrientedBox(10, 5, 3, 2, HALF_PI/3, 40, 40)
    assert ob.check_within_image() == IsWithinImage.YES
    _ = OrientedBox.from_points(ob.to_points(), 40, 40)
    _ = ob.to_box()
    _ = ob.to_polygon()
    assert ob == OrientedBox.from_numpy(ob.to_numpy(), 40, 40)
    _ = str(ob)

    assert ob == ob.clip()
    assert ob == ob.pad(1, 2, 3, 4).crop(3, 42, 1, 40)
    assert ob == ob.vertical_flip().vertical_flip()
    assert ob == ob.horizontal_flip().horizontal_flip()
    assert ob == ob.rotate(HALF_PI).rotate(-HALF_PI)
    assert ob == ob.rotate_right_angle(90).rotate_right_angle(270)
    assert ob.resize(80, 80) == ob.rescale(2, 2)
    assert ob == ob.transpose().transpose()


def test_obox_arr_simple():
    pass


def test_empty_oriented_box_array():
    a = OrientedBoxArray.from_oriented_boxes([], 40, 40)
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
