from typing import Iterable, List

import numpy as np

from .image import EmptyImage
from .point import Point
from .utils import IsWithinImage, rotate_point, rotate_point_right_angle
from .basic import ImageAnnotation
from .label import Label, Empty


class PointArray(ImageAnnotation):
    def __init__(
        self,
        x: Iterable[int],
        y: Iterable[int],
        img_w: int,
        img_h: int,
        label: Label = None,
    ):
        self._base_image = EmptyImage(img_w, img_h)
        self.x, self.y = np.array(x, dtype=int), np.array(y, dtype=int)
        if not isinstance(label, list):
            label = Empty() if label is None else label
            label = [label.copy() for _ in range(len(self))]
        super().__init__(label)
        self.check()

    def check(self):
        assert isinstance(self.label, list), f"self.label must be a list, got {type(self.label)}."
        assert len(self.label) == len(self), f"label length ({len(self.label)}) must equal len(self)({len(self)})."
        if len(self) > 0:
            assert all(isinstance(_l, type(self.label[0])) for _l in self.label),\
                "self.labels must be of same type."
        assert (self.x.ndim == 1), f"self.x and self.y must be 1-D array, got shape {self.x.shape}."
        assert (self.x.shape == self.y.shape),\
            f"self.x and self.y must be of same size, got x{self.x.shape} y{self.y.shape}."

    def is_within_image(self):
        return (self.x >= 0) & (self.x < self.img_w) & (self.y >= 0) & (self.y < self.img_h)

    def check_within_image(self):
        """
            If all points lie within the image, return YES.
            If all points lie outsiede the image, return NO.
            Else, return PARTIAL.
        """
        is_within = self.is_within_image()
        if np.all(is_within):
            return IsWithinImage.YES
        elif np.all(~is_within):
            return IsWithinImage.NO
        else:
            return IsWithinImage.PARTIAL

    def select(self, indices):
        """
        Select boxes by arbitrary indexing.

        Args:
            indices (Any): Any object that is legal for 1-D np.ndarray indexing

        Returns:
            PointArray
        """
        return PointArray(
            x=self.x[indices],
            y=self.y[indices],
            img_w=self.img_w,
            img_h=self.img_h,
            label=[self.label[i] for i in indices],
        )

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        return iter(self.to_points())

    def __getitem__(self, i):
        assert isinstance(i, int)
        if i < -len(self) or i >= len(self):
            raise IndexError
        return Point(self.x[i], self.y[i], self.img_w, self.img_h, self.label[i])

    @classmethod
    def from_points(cls, points: Iterable[Point], img_w: int = None, img_h: int = None):
        points = list(points)
        assert len(points) > 0 or (img_w is not None and img_h is not None),\
            "img_w and img_h cannot be infered from empty list."
        img_w = points[0].img_w if img_w is None else img_w
        img_h = points[0].img_h if img_h is None else img_h
        assert all([p.img_w == img_w for p in points])
        assert all([p.img_h == img_h for p in points])
        return cls(
            x=[p.x for p in points],
            y=[p.y for p in points],
            img_w=img_w,
            img_h=img_h,
            label=[p.label for p in points]
        )

    def to_points(self) -> List[Point]:
        return [Point(self.x[i], self.y[i], self.img_w, self.img_h, self.label[i]) for i in range(len(self))]

    @classmethod
    def from_numpy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        """
            array should be a np.ndarray shape = (N, 2). Columns represent x and y coordinates.
        """
        assert array.ndim == 2 and array.shape[1] == 2,\
            f"array must be of shape (N, 2), got {array.shape}"
        return cls(array[:, 0], array[:, 1], img_w, img_h, label)

    def to_numpy(self):
        """
            return the point coordinates as a (len(self), 2) ndarray. Inverse of from_numpy.
        """
        return np.stack([self.x, self.y], 1)

    @property
    def img_w(self):
        return self._base_image.width

    @property
    def img_h(self):
        return self._base_image.height

    def __repr__(self):
        s = f"PointArray(img_w={self.img_w}, img_h={self.img_h}, x=..., y=..., label=...):\n"
        for p in self:
            s += f"({p.x}, {p.y}), {p.label}\n"
        return s

    # transformations

    def _clip(self):
        is_within_image = np.arange(len(self))[self.is_within_image()]
        return PointArray(
            x=self.x[is_within_image],
            y=self.y[is_within_image],
            img_w=self.img_w,
            img_h=self.img_h,
            label=[self.label[i] for i in is_within_image],
        )

    def _pad(self, up, down, left, right, fill_value):
        timage = self._base_image.pad(up, down, left, right, fill_value)
        return PointArray(
            self.x + left,
            self.y + up,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        timage = self._base_image.crop(xmin, xmax, ymin, ymax)
        return PointArray(
            self.x - xmin,
            self.y - ymin,
            timage.img_w,
            timage.img_h,
            label=self.label,
        )

    def _horizontal_flip(self):
        return PointArray(
            self.img_w - self.x,
            self.y,
            self.img_w,
            self.img_h,
            label=self.label,
        )

    def _vertical_flip(self):
        return PointArray(
            self.x,
            self.img_h - self.y,
            self.img_w,
            self.img_h,
            label=self.label,
        )

    def _rotate(self, angle):
        timage = self._base_image.rotate(angle)
        rx, ry = rotate_point(self.img_w, self.img_h, angle, self.x, self.y)
        return PointArray(rx, ry, timage.img_w, timage.img_h, label=self.label)

    def _rotate_right_angle(self, right_angle):
        timage = self._base_image.rotate_right_angle(right_angle)
        rx, ry = rotate_point_right_angle(self.img_w, self.img_h, right_angle, self.x, self.y)
        return PointArray(rx, ry, timage.img_w, timage.img_h, label=self.label)

    def _resize(self, dst_w, dst_h):
        factor_x, factor_y = dst_w / self.img_w, dst_h / self.img_h
        return PointArray(
            self.x * factor_x,
            self.y * factor_y,
            dst_w,
            dst_h,
            label=self.label,
        )

    def _transpose(self):
        return PointArray(
            self.y, self.x, self.img_h, self.img_w, label=self.label
        )

    def __eq__(self, parr):
        if not isinstance(parr, PointArray):
            raise TypeError()
        return self.img_w == parr.img_w and\
            self.img_h == parr.img_h and\
            np.all(self.x == parr.x) and\
            np.all(self.y == parr.y) and\
            self.label == parr.label

    def __add__(self, pointarr_or_point):
        assert pointarr_or_point.img_w == self.img_w and pointarr_or_point.img_h == self.img_h
        if isinstance(pointarr_or_point, Point):
            return PointArray(
                x=np.concatenate([self.y, [pointarr_or_point.x]]),
                y=np.concatenate([self.x, [pointarr_or_point.y]]),
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label + [pointarr_or_point.label],
            )
        elif isinstance(pointarr_or_point, PointArray):
            return PointArray(
                x=np.concatenate([self.y, pointarr_or_point.x]),
                y=np.concatenate([self.x, pointarr_or_point.y]),
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label + pointarr_or_point.label,
            )
        else:
            raise TypeError()
