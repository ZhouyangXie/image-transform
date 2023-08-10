import numpy as np
import pickle as pkl

from ..point import Point
from ..polygon import Polygon
from ..mask import Mask
from ..utils import IsWithinImage, HALF_PI
from ..label import Scoped


class MaskScoped(Scoped):
    scope = ["a", "b", "c", "d"]


def test_mask_simple():
    mask = Mask(
        mask=np.array([[0, 1], [2, 3]], np.uint8),
        label=MaskScoped("a")
    )
    assert mask.unique_labels == ["a", "b", "c"]
    assert mask.img_w == 2 and mask.img_h == 2
    assert mask.check_within_image() == IsWithinImage.YES
    assert len(mask.scope) == 4
    _ = Mask.from_polygons([
        Polygon([
            Point(0, 0, 10, 10), Point(0, 5, 10, 10), Point(5, 5, 10, 10), Point(5, 0, 10, 10)
        ], 10, 10, MaskScoped('a')),
        Polygon([
            Point(5, 5, 10, 10), Point(9, 5, 10, 10), Point(9, 9, 10, 10), Point(5, 9, 10, 10)
        ], 10, 10, MaskScoped('b')),
    ])

    assert mask == Mask.from_numpy(mask.to_numpy(), MaskScoped("d"))
    assert mask == mask.clip()
    assert mask.pad(1, 0, 0, 1) == Mask(
        np.array([[0, 0, 0], [0, 1, 0], [2, 3, 0]], np.int32), MaskScoped("c"))
    assert mask == mask.pad(1, 2, 3, 4).crop(3, 4, 1, 2)
    assert mask == mask.horizontal_flip().horizontal_flip()
    assert mask == mask.vertical_flip().vertical_flip()
    _ = mask.rotate(HALF_PI/4)
    assert mask.rotate_right_angle(0) == mask.rotate_right_angle(90).rotate_right_angle(270)
    assert mask.rotate_right_angle(0) == mask.rotate_right_angle(180).rotate_right_angle(180)
    assert mask == mask.rescale(2, 3).resize(2, 2)
    assert mask == mask.transpose().transpose()
    bytes = pkl.dumps(mask)
    _mask = pkl.loads(bytes)
    assert mask == _mask

    mask = Mask(mask=(np.random.rand(100, 100) > 0.05).astype(np.uint8))
    bytes = pkl.dumps(mask)
    _mask = pkl.loads(bytes)
    assert mask == _mask
