import numpy as np
import pytest

from ..image import Image
from ..point import Point
from ..box import Box
from ..box_array import BoxArray
from ..oriented_box import OrientedBox
from ..mask import Mask
from ..composite import Composite
from ..utils import IsWithinImage, HALF_PI


def test_composite():
    with pytest.raises(AssertionError):
        comp = Composite([])
    comp = Composite([], 100, 100)
    assert len(comp) == 0
    assert len(comp.flatten()) == 0
    assert len(comp.compact()) == 3
    assert comp.images == []
    assert comp.image is None
    comp = Composite([
        Image(np.zeros((40, 40)), "a"),
        Point(10, 10, 40, 40, "b"),
        Box(20, 30, 30, 35, 40, 40, "c"),
        OrientedBox(25, 35, 6, 6, HALF_PI, 40, 40),
        Mask(np.ones((40, 40), np.int32), ["a"])
    ])
    assert len(comp) == 5
    assert len(list(iter(comp))) == 5
    assert isinstance(comp[3], OrientedBox)

    assert comp.check_within_image() == IsWithinImage.YES
    _ = comp.to_numpy()
    _ = str(comp)
    assert comp.images[0] == comp.image
    assert comp.num_image == 1
    assert set(comp.unique_labels) == set(["a", "b", "c", None])
    _ = comp.clip()
    _ = comp.pad(1, 1, 1, 1)
    _ = comp.crop(10, 30, 10, 30)
    _ = comp.horizontal_flip()
    _ = comp.vertical_flip()
    _ = comp.rotate(HALF_PI/4)
    _ = comp.rotate_right_angle(90)
    _ = comp.rescale(1)
    _ = comp.resize(20, 20)
    _ = comp.transpose()
    _ = comp.filter_within_image(IsWithinImage.YES)
    assert len(comp.filter_type([Box])) == 1
    _ = comp.compact()

    comp = Composite([
        Image(np.zeros((40, 40)), "a"),
        Point(10, 10, 40, 40),
        Point(10, 10, 40, 40),
        Point(10, 10, 40, 40,),
        BoxArray([20, 20], [30, 30], [30, 30], [35, 35], 40, 40),
        Mask(np.ones((40, 40), np.int32))
    ])
    _comp = comp.compact()
    assert len(_comp) == 5
    _comp = comp.flatten()
    assert len(_comp) == 7

    del comp[3]
    assert len(comp) == 5
    comp_s = comp[2:4]
    assert isinstance(comp_s, Composite)
    assert len(comp_s) == 2
    comp_s.append(comp_s[0])
    comp_s.extend(comp_s)
