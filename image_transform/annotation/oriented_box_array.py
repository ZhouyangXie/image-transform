from typing import List, Iterable
from itertools import product

import numpy as np

from .basic import ImageAnnotation
from .image import EmptyImage
from .point import Point
from .oriented_box import OrientedBox
from .label import Label, Empty
from .utils import IsWithinImage, HALF_PI, PI, ONE_AND_HALF_PI, normalize,\
    rotate_point, rotate_point_right_angle, nms


class OrientedBoxArray(ImageAnnotation):
    def __init__(
        self,
        x: Iterable[int],
        y: Iterable[int],
        w: Iterable[int],
        h: Iterable[int],
        theta: Iterable[float],
        img_w: int,
        img_h: int,
        label: Label = None,
    ) -> None:
        """
        Args:
            x (Iterable[int]): x-coordinates of the box centers
            y (Iterable[int]): y-coordinates of the box centers
            w (Iterable[int]): box widths
            h (Iterable[int]): box heights
            theta (Iterable[float]): the orientations of the boxes
            img_w (int): the width of the base image
            img_h (int): the height of the base image
        """
        self.x = np.array(x, dtype=int)
        self.y = np.array(y, dtype=int)
        self.w = np.array(w, dtype=int)
        self.w[self.w == 0] = 1
        self.h = np.array(h, dtype=int)
        self.h[self.h == 0] = 1
        self.theta = normalize(np.array(theta, dtype=float))
        self._base_image = EmptyImage(img_w, img_h)
        if not isinstance(label, list):
            label = Empty() if label is None else label
            label = [label.copy() for _ in range(len(self))]
        super().__init__(label)
        self.check()

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        return iter(self.to_oriented_boxes())

    def __getitem__(self, i):
        assert isinstance(i, int)
        if i < -len(self) or i >= len(self):
            raise IndexError
        return OrientedBox(
            self.x[i],
            self.y[i],
            self.w[i],
            self.h[i],
            self.theta[i],
            self.img_w,
            self.img_h,
            self.label[i],
        )

    def check(self):
        assert isinstance(self.label, list)
        assert len(self.label) == len(self), f"label length ({len(self.label)}) must equal len(self)({len(self)})"
        if len(self) > 0:
            assert all(isinstance(_l, type(self.label[0])) for _l in self.label),\
                "self.labels must be of same type."
        assert np.all(self.width > 0), f"width must be greater than 0, got {self.width}"
        assert np.all(self.height > 0), f"height must be greater than 0, got {self.height}"

    def is_within_image(self):
        return (self.xmin >= 0) & (self.xmax < self.img_w) & (self.ymin >= 0) & (self.ymax < self.img_h)

    def is_outside_image(self):
        return (self.xmin >= self.img_w) | (self.xmax < 0) | (self.ymin >= self.img_h) | (self.ymax < 0)

    def check_within_image(self):
        """
        If every box is within the image, return YES.
        If every box is outside the image, return NO.
        Else, return PARTIAL.
        """
        is_within = self.is_within_image()
        is_outside = self.is_outside_image()
        if np.all(is_within):
            return IsWithinImage.YES
        elif np.all(is_outside):
            return IsWithinImage.NO
        else:
            return IsWithinImage.PARTIAL

    @classmethod
    def from_oriented_boxes(cls, oriented_boxes: Iterable[OrientedBox], img_w: int = None, img_h: int = None):
        oriented_boxes = list(oriented_boxes)
        assert len(oriented_boxes) > 0 or (img_w is not None and img_h is not None),\
            "img_w and img_h cannot be infered from empty list."
        img_w = oriented_boxes[0].img_w if img_w is None else img_w
        img_h = oriented_boxes[0].img_h if img_h is None else img_h
        assert all([b.img_w == img_w for b in oriented_boxes])
        assert all([b.img_h == img_h for b in oriented_boxes])
        return cls(
            x=[b.x for b in oriented_boxes],
            y=[b.y for b in oriented_boxes],
            w=[b.w for b in oriented_boxes],
            h=[b.h for b in oriented_boxes],
            theta=[b.theta for b in oriented_boxes],
            img_w=img_w,
            img_h=img_h,
            label=[b.label for b in oriented_boxes],
        )

    def to_oriented_boxes(self) -> List[OrientedBox]:
        return [
            OrientedBox(
                x=self.x[i],
                y=self.y[i],
                w=self.w[i],
                h=self.h[i],
                theta=self.theta[i],
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label[i],
            )
            for i in range(len(self))
        ]

    def select(self, indices):
        """
        Select boxes by arbitrary indexing.

        Args:
            indices (Any): Any object that is legal for 1-D np.ndarray indexing

        Returns:
            OrientedBoxArray
        """
        return OrientedBoxArray(
            x=self.x[indices],
            y=self.y[indices],
            w=self.w[indices],
            h=self.h[indices],
            theta=self.theta[indices],
            img_w=self.img_w,
            img_h=self.img_h,
            label=[self.label[i] for i in indices],
        )

    def nms(self, iou_threshold=0.5):
        """
            Perform Non-maximum Suppression.
            Assume that the box array has been sorted by confidence in descending order.

        Args:
            iou_threshold (float, optional): IoU threshold to determine duplicate. Defaults to 0.5.

        Returns:
            OrientedBoxArray: duplicate removed box array
        """
        iou_matrix = self.iou(self)
        duplicate_matrix = iou_matrix >= iou_threshold
        preserved = nms(duplicate_matrix)
        return self.select(preserved.nonzero()[0])

    # properties

    @property
    def width(self):
        return self.w

    @property
    def height(self):
        return self.h

    @property
    def horizontal_width(self):
        return np.abs(self.h * np.sin(self.theta)) + np.abs(self.w * np.cos(self.theta))

    @property
    def vertical_height(self):
        return np.abs(self.w * np.sin(self.theta)) + np.abs(self.h * np.cos(self.theta))

    @property
    def center_x(self):
        return self.x

    @property
    def center_y(self):
        return self.y

    @property
    def aspect_ratio(self):
        return self.width / self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def xmin(self):
        return self.center_x - self.horizontal_width / 2

    @property
    def xmax(self):
        return self.center_x + self.horizontal_width / 2

    @property
    def ymin(self):
        return self.center_y - self.vertical_height / 2

    @property
    def ymax(self):
        return self.center_y + self.vertical_height / 2

    @property
    def img_w(self):
        return self._base_image.img_w

    @property
    def img_h(self):
        return self._base_image.img_h

    def __repr__(self):
        s = f"OrientedBox(img_w={self.img_w}, img_h={self.img_h}, x=..., y=..., w=..., h=..., theta=..., label=...):\n"
        for b in self:
            s += f"({b.x}, {b.y}, {b.w}, {b.h}, {b.theta}), {b.label},\n"
        return s

    @classmethod
    def from_numpy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.ndim == 2 and array.shape[1] == 5
        return cls(array[:, 0], array[:, 1], array[:, 2], array[:, 3], array[:, 4], img_w, img_h, label)

    def to_numpy(self):
        return np.stack([self.x, self.y, self.w, self.h, self.theta], 1)

    # transforms

    def _clip(self):
        is_within = self.is_within_image()
        is_outside = self.is_outside_image()
        is_partial = ~(is_within | is_outside)
        if np.all(is_within) or np.all(is_outside):
            return self
        elif np.any(is_partial):
            raise NotImplementedError("Some oriented boxes are partially within the image. Cannot be clipped for now.")
        else:
            return OrientedBoxArray(
                x=self.x[is_within],
                y=self.y[is_within],
                w=self.w[is_within],
                h=self.h[is_within],
                theta=self.theta[is_within],
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label,
            )

    def _pad(self, up, down, left, right, fill_value=None):
        timage = self._base_image.pad(up, down, left, right, fill_value)
        return OrientedBoxArray(
            x=self.x + left,
            y=self.y + up,
            w=self.w,
            h=self.h,
            theta=self.theta,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        timage = self._base_image.crop(xmin, xmax, ymin, ymax)
        return OrientedBoxArray(
            x=self.x - xmin,
            y=self.y - ymin,
            w=self.w,
            h=self.h,
            theta=self.theta,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _horizontal_flip(self):
        return OrientedBoxArray(
            x=self.img_w - self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            theta=PI - self.theta,
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    def _vertical_flip(self):
        return OrientedBoxArray(
            x=self.x,
            y=self.img_h - self.y,
            w=self.w,
            h=self.h,
            theta=-self.theta,
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    def _rotate(self, angle):
        rx, ry = rotate_point(self.img_w, self.img_h, angle, self.x, self.y)
        timage = self._base_image.rotate(angle)
        return OrientedBoxArray(
            x=rx,
            y=ry,
            w=self.w,
            h=self.h,
            theta=self.theta + angle,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _rotate_right_angle(self, right_angle):
        rx, ry = rotate_point_right_angle(
            self.img_w, self.img_h, right_angle, self.x, self.y
        )
        if right_angle == 90:
            angle = HALF_PI
        elif right_angle == 180:
            angle = PI
        elif right_angle == 270:
            angle = ONE_AND_HALF_PI
        else:
            angle = 0.0

        timage = self._base_image.rotate_right_angle(right_angle)
        return OrientedBoxArray(
            x=rx,
            y=ry,
            w=self.w,
            h=self.h,
            theta=self.theta + angle,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _resize(self, dst_w, dst_h):
        """
        Resize box and image.
        Note that after resizing/rescaling, a non-horizontal box will be a parallelogram,
        so we only keep the enclosing oriented box of this parallelogram
        (by OrientedBox.from_points)
        """
        factor_x, factor_y = dst_w / self.img_w, dst_h / self.img_h
        resized_oriented_boxes = []
        for oriented_box in self:
            p0, p1, p2, p3 = oriented_box.to_points()
            resized_oriented_box = OrientedBox.from_points(
                points=[
                    Point(
                        p0.x * factor_x,
                        p0.y * factor_y,
                        dst_w,
                        dst_h,
                        label=oriented_box.label,
                    ),
                    Point(
                        p1.x * factor_x,
                        p1.y * factor_y,
                        dst_w,
                        dst_h,
                        label=oriented_box.label,
                    ),
                    Point(
                        p2.x * factor_x,
                        p2.y * factor_y,
                        dst_w,
                        dst_h,
                        label=oriented_box.label,
                    ),
                    Point(
                        p3.x * factor_x,
                        p3.y * factor_y,
                        dst_w,
                        dst_h,
                        label=oriented_box.label,
                    ),
                ],
                img_w=dst_w,
                img_h=dst_h,
            )
            resized_oriented_boxes.append(resized_oriented_box)

        return OrientedBoxArray.from_oriented_boxes(resized_oriented_boxes, dst_w, dst_h)

    def _transpose(self):
        return OrientedBoxArray(
            x=self.y,
            y=self.x,
            w=self.w,
            h=self.h,
            theta=HALF_PI - self.theta,
            img_w=self.img_h,
            img_h=self.img_w,
            label=self.label,
        )

    # relations

    def intersection_area(self, other) -> np.ndarray:
        """
            Compute the intersection area between each pair of boxes.

        Args:
            other (OrientedBoxArray): another OrientedBoxArray

        Returns:
            np.ndarray: shape (len(self), len(other)) float ndarray
        """
        assert isinstance(other, OrientedBoxArray),\
            "currently I/U/IoU computation is only supported for other OrientedBoxArray"
        assert self.img_w == other.img_w and self.img_h == other.img_h,\
            f"must have save base image shape self({self.img_w}, {self.img_h}) v.s. box({other.img_w}, {other.img_h})"
        inter_area = np.zeros((len(self), len(other)), dtype=float)
        for (i, ob_i), (j, ob_j) in product(enumerate(self), enumerate(other)):
            inter_area[i, j] = ob_i.intersection_area(ob_j)

        return inter_area

    def union_area(self, other) -> np.ndarray:
        """
            Compute the union area between each pair of boxes.

        Args:
            other (OrientedBoxArray): another OrientedBoxArray

        Returns:
            np.ndarray: shape (len(self), len(other)) float ndarray
        """
        return np.add.outer(self.area, other.area) - self.intersection_area(other)

    def iou(self, other) -> np.ndarray:
        """
            Compute the IoU between each pair of boxes.

        Args:
            other (BoxArray): another BoxArray

        Returns:
            np.ndarray: shape (len(self), len(other)) float ndarray
        """
        _EPS = 1e-10
        inter_area = self.intersection_area(other)
        union_area = np.add.outer(self.area, other.area) - inter_area + _EPS
        return inter_area / union_area

    def __eq__(self, boxarr):
        if not isinstance(boxarr, OrientedBoxArray):
            raise TypeError()
        return self.img_w == boxarr.img_w and\
            self.img_h == boxarr.img_h and\
            np.all(self.x == boxarr.x) and\
            np.all(self.y == boxarr.y) and\
            np.all(self.w == boxarr.w) and\
            np.all(self.h == boxarr.h) and\
            np.all((self.theta - boxarr.theta) < 1e-5) and\
            self.label == boxarr.label

    def __add__(self, boxarr_or_box):
        assert boxarr_or_box.img_w == self.img_w and boxarr_or_box.img_h == self.img_h
        if isinstance(boxarr_or_box, OrientedBox):
            return OrientedBoxArray(
                x=np.concatenate([self.y, [boxarr_or_box.x]]),
                y=np.concatenate([self.x, [boxarr_or_box.y]]),
                w=np.concatenate([self.w, [boxarr_or_box.w]]),
                h=np.concatenate([self.h, [boxarr_or_box.h]]),
                theta=np.concatenate([self.theta, [boxarr_or_box.theta]]),
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label + [boxarr_or_box.label],
            )
        elif isinstance(boxarr_or_box, OrientedBoxArray):
            return OrientedBoxArray(
                x=np.concatenate([self.y, boxarr_or_box.x]),
                y=np.concatenate([self.x, boxarr_or_box.y]),
                w=np.concatenate([self.w, boxarr_or_box.w]),
                h=np.concatenate([self.h, boxarr_or_box.h]),
                theta=np.concatenate([self.theta, boxarr_or_box.theta]),
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label + boxarr_or_box.label,
            )
        else:
            raise TypeError()
