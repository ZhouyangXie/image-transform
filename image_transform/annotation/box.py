import numpy as np

from .image import EmptyImage
from .point import Point
from .polygon import Polygon
from .utils import IsWithinImage, ONE_AND_HALF_PI
from .basic import ImageAnnotation
from .label import Label


class Box(ImageAnnotation):
    def __init__(
        self, xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
        img_w: int,
        img_h: int,
        label: Label = None,
    ) -> None:
        """
        Args:
            xmin (int): left bound index, inclusive. 0 <= xmin < xmax < img_w
            xmax (int): right bound index, inclusive.
            ymin (int): upper bound index, inclusive. 0 <= ymin < ymax < img_h
            ymax (int): lower bound index, inclusive
            img_w (int): the width of the base image
            img_h (int): the height of the base image
        """
        super().__init__(label)
        self._base_image = EmptyImage(img_w, img_h)
        self.xmin = int(xmin)
        self.xmax = int(xmax)
        self.ymin = int(ymin)
        self.ymax = int(ymax)
        self.check()

    def check(self):
        assert self.xmax > self.xmin,\
            f"xmax({self.xmax}) should be greater than xmin({self.xmin})"
        assert self.ymax > self.ymin,\
            f"ymax({self.ymax}) should be greater than ymin({self.ymin})"

    def check_within_image(self):
        if (self.xmin >= self.img_w or self.xmax < 0 or self.ymin >= self.img_h or self.ymax < 0):
            return IsWithinImage.NO
        elif (self.xmax >= self.img_w or self.xmin < 0 or self.ymax >= self.img_h or self.ymin < 0):
            return IsWithinImage.PARTIAL
        else:
            return IsWithinImage.YES

    @classmethod
    def from_corner_points(cls, top_left_point, bottom_right_point, img_w: int = None, img_h: int = None):
        """
        create a box from top-left and bottom-right points
        """
        img_w = img_w if img_w is not None else top_left_point.img_w
        img_h = img_h if img_h is not None else top_left_point.img_h
        return cls(
            xmin=top_left_point.x,
            ymin=top_left_point.y,
            xmax=bottom_right_point.x,
            ymax=bottom_right_point.y,
            img_w=img_w,
            img_h=img_h,
            label=top_left_point.label
        )

    def to_points(self):
        """
        top-left, top-right, bottom-left, bottom-right points
        """
        return [
            Point(self.xmin, self.ymin, self.img_w, self.img_h, self.label),
            Point(self.xmax, self.ymin, self.img_w, self.img_h, self.label),
            Point(self.xmin, self.ymax, self.img_w, self.img_h, self.label),
            Point(self.xmax, self.ymax, self.img_w, self.img_h, self.label),
        ]

    def to_polygon(self):
        tl, tr, bl, br = self.to_points()
        return Polygon([tl, tr, br, bl], self.img_w, self.img_h, label=self.label)

    def to_oriented_box(self):
        """
           Return OrientedBox are oriented to 1.5 pi (up-side).
        """
        from .oriented_box import OrientedBox
        return OrientedBox(
            x=self.center_x,
            y=self.center_y,
            w=self.width,
            h=self.height,
            theta=ONE_AND_HALF_PI,
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    @classmethod
    def from_numpy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        """
            array must be of shape (4, ), representing xmin, xmax, ymin, ymax.
        """
        assert array.shape == (4, ), f"array must be a np.ndarray shape=(4, ), got {array.shape}."
        return cls(array[0], array[1], array[2], array[3], img_w, img_h, label)

    def to_numpy(self):
        """
            inverse of from_numpy()
        """
        return np.array([self.xmin, self.xmax, self.ymin, self.ymax])

    @classmethod
    def from_numpy_as_xyxy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.shape == (4, ), f"array must be a np.ndarray shape=(4, ), got {array.shape}."
        xmin, ymin, xmax, ymax = array[0], array[1], array[2], array[3]
        return cls(xmin, xmax, ymin, ymax, img_w, img_h, label)

    def to_numpy_as_xyxy(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    @classmethod
    def from_numpy_as_cxcywh(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.shape == (4, ), f"array must be a np.ndarray shape=(4, ), got {array.shape}."
        cx, cy, w, h = array[0], array[1], array[2], array[3]
        half_w, half_h = w/2, h/2
        xmin, xmax, ymin, ymax = cx - half_w, cx + half_w, cy - half_h, cy + half_h
        return cls(xmin, xmax, ymin, ymax, img_w, img_h, label)

    def to_numpy_as_cxcywh(self):
        return np.array([self.center_x, self.center_y, self.width, self.height])

    @classmethod
    def from_numpy_as_xywh(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.shape == (4, ), f"array must be a np.ndarray shape=(4, ), got {array.shape}."
        xmin, ymin, w, h = array[0], array[1], array[2], array[3]
        xmax, ymax = xmin + w, ymin + h
        return cls(xmin, xmax, ymin, ymax, img_w, img_h, label)

    def to_numpy_as_xywh(self):
        return np.array([self.xmin, self.ymin, self.width, self.height])

    # properties

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def center_x(self):
        return (self.xmin + self.xmax) // 2

    @property
    def center_y(self):
        return (self.ymin + self.ymax) // 2

    @property
    def aspect_ratio(self):
        return self.width / self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def img_w(self):
        return self._base_image.width

    @property
    def img_h(self):
        return self._base_image.height

    def __repr__(self):
        return (
            f"Box(img_w={self.img_w}, img_h={self.img_h}, xmin={self.xmin}, "
            f"xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax}, label={self.label}"
        )

    # transforms

    def _clip(self):
        if self.check_within_image() == IsWithinImage.PARTIAL:
            return Box(
                xmin=max(0, self.xmin),
                xmax=min(self.img_w - 1, self.xmax),
                ymin=max(0, self.ymin),
                ymax=min(self.img_h - 1, self.ymax),
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label,
            )
        else:
            return self

    def _pad(self, up, down, left, right, fill_value):
        timage = self._base_image.pad(up, down, left, right, fill_value)
        return Box(
            xmin=self.xmin + left,
            xmax=self.xmax + left,
            ymin=self.ymin + up,
            ymax=self.ymax + up,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        timage = self._base_image.crop(xmin, xmax, ymin, ymax)
        return Box(
            xmin=self.xmin - xmin,
            xmax=self.xmax - xmin,
            ymin=self.ymin - ymin,
            ymax=self.ymax - ymin,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _horizontal_flip(self):
        timage = self._base_image.horizontal_flip()
        return Box(
            xmin=self.img_w - self.xmax,
            xmax=self.img_w - self.xmin,
            ymin=self.ymin,
            ymax=self.ymax,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _vertical_flip(self):
        timage = self._base_image.horizontal_flip()
        return Box(
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.img_h - self.ymax,
            ymax=self.img_h - self.ymin,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _rotate(self, angle):
        """
        Rotate the box with the image by some degree. This function returns an OrientedBox.
        """
        return self.to_oriented_box().rotate(angle)

    def _rotate_right_angle(self, rotate_right_angle):
        timage = self._base_image.rotate_right_angle(rotate_right_angle)
        if rotate_right_angle == 90:
            return Box(
                xmin=self.img_h - self.ymax,
                xmax=self.img_h - self.ymin,
                ymin=self.xmin,
                ymax=self.xmax,
                img_w=timage.img_w,
                img_h=timage.img_h,
                label=self.label,
            )
        elif rotate_right_angle == 180:
            return Box(
                xmin=self.img_w - self.xmax,
                xmax=self.img_w - self.xmin,
                ymin=self.img_h - self.ymax,
                ymax=self.img_h - self.ymin,
                img_w=timage.img_w,
                img_h=timage.img_h,
                label=self.label,
            )
        elif rotate_right_angle == 270:
            return Box(
                xmin=self.ymin,
                xmax=self.ymax,
                ymin=self.img_w - self.xmax,
                ymax=self.img_w - self.xmin,
                img_w=timage.img_w,
                img_h=timage.img_h,
                label=self.label,
            )
        else:
            return self

    def _resize(self, dst_w, dst_h):
        factor_x, factor_y = dst_w / self.img_w, dst_h / self.img_h
        return Box(
            xmin=self.xmin * factor_x,
            xmax=self.xmax * factor_x,
            ymin=self.ymin * factor_y,
            ymax=self.ymax * factor_y,
            img_w=dst_w,
            img_h=dst_h,
            label=self.label,
        )

    def _transpose(self):
        timage = self._base_image.transpose()
        return Box(
            xmin=self.ymin,
            xmax=self.ymax,
            ymin=self.xmin,
            ymax=self.xmax,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    # relations

    def __eq__(self, box):
        if not isinstance(box, Box):
            raise TypeError()
        return self.img_w == box.img_w and\
            self.img_h == box.img_h and\
            self.xmin == box.xmin and\
            self.xmax == box.xmax and\
            self.ymin == box.ymin and\
            self.ymax == box.ymax and\
            self.label == box.label

    def intersection_area(self, other) -> int:
        assert isinstance(other, Box),\
            "currently I/U/IoU computation is only supported for Box-Box or BoxArray-BoxArray"
        assert self.img_w == other.img_w and self.img_h == other.img_h,\
            f"must have save base image shape self({self.img_w}, {self.img_h}) v.s. box({other.img_w}, {other.img_h})"
        inter_xmin = max(self.xmin, other.xmin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymin = max(self.ymin, other.ymin)
        inter_ymax = min(self.ymax, other.ymax)

        if inter_xmin > inter_xmax or inter_ymin > inter_ymax:
            return 0
        else:
            return (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    def union_area(self, other) -> int:
        assert isinstance(other, Box), "currently I/U/IoU computation is only supported for same type."
        assert self.img_w == other.img_w and self.img_h == other.img_h,\
            f"must have save base image shape self({self.img_w}, {self.img_h}) vs box({other.img_w}, {other.img_h})"
        return self.area + other.area - self.intersection_area(other)

    def iou(self, other) -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union
