import numpy as np
from typing import Tuple, Union

from .image import EmptyImage
from .utils import IsWithinImage, rotate_point, rotate_point_right_angle
from .basic import ImageAnnotation
from .label import Label


class Point(ImageAnnotation):
    def __init__(self, x: int, y: int, img_w: int, img_h: int, label: Label = None):
        super().__init__(label)
        self._base_image, self.x, self.y = EmptyImage(img_w, img_h), int(x), int(y)
        self.check()

    def check_within_image(self):
        if (self.x < 0 or self.x >= self._base_image.width or self.y < 0 or self.y >= self._base_image.height):
            return IsWithinImage.NO
        else:
            return IsWithinImage.YES

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, img_w={self.img_w}, img_h={self.img_h}), label={self.label}"

    @classmethod
    def from_numpy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        """
            array should be a np.ndarray shape = (2), representing x and y coordinates.
        """
        assert array.shape == (2, ), f"array must be of shape (2, ), got {array.shape}"
        return cls(array[0], array[1], img_w, img_h, label)

    def to_numpy(self):
        """
            Return the point coordinates as a shape = (2, ) ndarray.
            Inverse of from_numpy().
        """
        return np.array([self.x, self.y])

    # properties
    @property
    def img_w(self):
        return self._base_image.width

    @property
    def img_h(self):
        return self._base_image.height

    # transformations

    def _clip(self):
        return self

    def _pad(self, up, down, left, right, fill_value):
        timage = self._base_image.pad(up, down, left, right, fill_value)
        return Point(
            x=self.x + left,
            y=self.y + up,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        timage = self._base_image.crop(xmin, xmax, ymin, ymax)
        return Point(
            x=self.x - xmin,
            y=self.y - ymin,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _horizontal_flip(self):
        timage = self._base_image.horizontal_flip()
        return Point(
            x=self.img_w - self.x,
            y=self.y,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _vertical_flip(self):
        timage = self._base_image.vertical_flip()
        return Point(
            x=self.x,
            y=self.img_h - self.y,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _rotate(self, angle):
        timage = self._base_image.rotate(angle)
        rx, ry = rotate_point(self.img_w, self.img_h, angle, self.x, self.y)
        return Point(x=rx, y=ry, img_w=timage.img_w, img_h=timage.img_h, label=self.label)

    def _rotate_right_angle(self, right_angle):
        timage = self._base_image.rotate_right_angle(right_angle)
        rx, ry = rotate_point_right_angle(
            self.img_w, self.img_h, right_angle, self.x, self.y
        )
        return Point(x=rx, y=ry, img_w=timage.img_w, img_h=timage.img_h, label=self.label)

    def _resize(self, dst_w, dst_h):
        factor_x, factor_y = dst_w / self.img_w, dst_h / self.img_h
        return Point(
            x=self.x * factor_x,
            y=self.y * factor_y,
            img_w=dst_w,
            img_h=dst_h,
            label=self.label,
        )

    def _transpose(self):
        return Point(
            x=self.y,
            y=self.x,
            img_w=self.img_h,
            img_h=self.img_w,
            label=self.label
        )

    # relations

    def __eq__(self, p):
        if isinstance(p, Point):
            if p.img_w != self.img_w or p.img_h != self.img_h:
                return False
        elif isinstance(p, (tuple, list)) and len(p) == 2:
            p = Point(p[0], p[1], self.img_w, self.img_h, self.label)
        else:
            return False
            # raise TypeError("Point can be compared with Point or tuple[int, int]")

        return p.x == self.x and p.y == self.y and self.label == p.label


def line_intersection(
    segment_a: Tuple[Point, Point], segment_b: Tuple[Point, Point]
) -> Union[Point, None]:
    """
    Find the intersection(Point) of two line segment if any(else None)

    Args:
        segment_a (Tuple[Point, Point])
        segment_b (Tuple[Point, Point])

    Returns:
        Union[None, Point]
    """
    def _tan(diff_x: int, diff_y: int):
        if diff_x == 0:
            return np.inf if diff_y >= 0 else - np.inf
        else:
            return diff_y/diff_x

    pa1, pa2 = segment_a
    pb1, pb2 = segment_b
    assert pa1.img_w == pa2.img_w == pb1.img_w == pb2.img_w
    assert pa1.img_h == pa2.img_h == pb1.img_h == pb2.img_h

    if pa1 == pb1:
        return pa1

    tan_a, tan_b = _tan(pa1.x - pa2.x, pa1.y - pa2.y), _tan(pb1.x - pb2.x, pb1.y - pb2.y)
    atan_a, atan_b = _tan(pa1.y - pa2.y, pa1.x - pa2.x), _tan(pb1.y - pb2.y, pb1.x - pb2.x)

    # handle parallel segments
    # cuz point coordinates are int, tan/atan are exactly same
    # we assume that the start points (pa1 and pb1) belong to the segment, while the end poits do not
    if tan_a == tan_b or atan_a == atan_b:
        return None

    if np.isinf(tan_a):  # pa1.x == pa2.x
        x = pa1.x
    elif np.isinf(tan_b):  # pb1.x == pb2.x
        x = pb1.x
    else:
        x = ((tan_a * pa1.x - tan_b * pb1.x) - (pa1.y - pb1.y)) / (tan_a - tan_b)

    if np.isinf(atan_a):  # pa1.y == pa2.y
        y = pa1.y
    elif np.isinf(atan_b):  # pb1.y == pb2.y
        y = pb1.y
    else:
        y = ((atan_a * pa1.y - atan_b * pb1.y) - (pa1.x - pb1.x)) / (atan_a - atan_b)

    if pa1.x > pa2.x and (x > pa1.x or x <= pa2.x):
        return None
    elif pa1.x < pa2.x and (x < pa1.x or x >= pa2.x):
        return None
    elif pa1.y > pa2.y and (y > pa1.y or y <= pa2.y):
        return None
    elif pa1.y < pa2.y and (y < pa1.y or y >= pa2.y):
        return None
    elif pb1.x > pb2.x and (x > pb1.x or x <= pb2.x):
        return None
    elif pb1.x < pb2.x and (x < pb1.x or x >= pb2.x):
        return None
    elif pb1.y > pb2.y and (y > pb1.y or y <= pb2.y):
        return None
    elif pb1.y < pb2.y and (y < pb1.y or y >= pb2.y):
        return None
    else:
        return Point(x=int(x), y=int(y), img_w=pa1.img_w, img_h=pa1.img_h)
