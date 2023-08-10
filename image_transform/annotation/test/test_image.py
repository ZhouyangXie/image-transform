import numpy as np

from ..image import EmptyImage, Image
from ..utils import IsWithinImage, HALF_PI


def test_image_simple():
    image = Image(np.array([[0, 1], [2, 3]], np.uint8))
    assert image.img_w == 2 and image.img_h == 2
    assert image.check_within_image() == IsWithinImage.YES
    assert image == Image.from_numpy(image.to_numpy())
    assert image == Image.from_pil(image.to_pil())
    assert image.to_empty_image() == EmptyImage(2, 2, 1, np.uint8)
    _ = str(image)
    assert image == image.clip()
    assert image.pad(1, 0, 0, 1) == Image(np.array([[0, 0, 0], [0, 1, 0], [2, 3, 0]], np.uint8))
    assert image == image.pad(1, 2, 3, 4).crop(3, 4, 1, 2)
    assert image == image.horizontal_flip().horizontal_flip()
    assert image == image.vertical_flip().vertical_flip()
    _ = image.rotate(HALF_PI/4)
    assert image.rotate_right_angle(0) == image.rotate_right_angle(90).rotate_right_angle(270)
    assert image.rotate_right_angle(0) == image.rotate_right_angle(180).rotate_right_angle(180)
    assert image == image.rescale(2, 3).resize(2, 2)
    assert image == image.transpose().transpose()
