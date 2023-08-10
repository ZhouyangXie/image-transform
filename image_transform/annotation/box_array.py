import numpy as np
from typing import Iterable, List
from .image import EmptyImage
from .point_array import PointArray
from .box import Box
from .utils import IsWithinImage, ONE_AND_HALF_PI, nms
from .basic import ImageAnnotation
from .label import Label, Empty


class BoxArray(ImageAnnotation):
    def __init__(
        self,
        xmin: Iterable[int],
        xmax: Iterable[int],
        ymin: Iterable[int],
        ymax: Iterable[int],
        img_w: int,
        img_h: int,
        label: Label = None,
    ) -> None:
        """
        Args:
            xmin (Iterable[int]): left bound indices, inclusive. 0 <= xmin < xmax < img_w
            xmax (Iterable[int]): right bound indices, inclusive.
            ymin (Iterable[int]): upper bound indices, inclusive. 0 <= ymin < ymax < img_h
            ymax (Iterable[int]): lower bound indices, inclusive
            img_w (int): the width of the base image
            img_h (int): the height of the base image
        """
        self._base_image = EmptyImage(img_w, img_h)
        self.xmin = np.array(xmin, dtype=int)
        self.xmax = np.array(xmax, dtype=int)
        self.xmax[self.xmin == self.xmax] += 1
        self.ymin = np.array(ymin, dtype=int)
        self.ymax = np.array(ymax, dtype=int)
        self.ymax[self.ymin == self.ymax] += 1
        if not isinstance(label, list):
            label = Empty() if label is None else label
            label = [label.copy() for _ in range(len(self))]
        super().__init__(label)
        self.check()

    def __len__(self):
        return len(self.xmin)

    def __iter__(self):
        return iter(self.to_boxes())

    def __getitem__(self, i):
        assert isinstance(i, int)
        if i < -len(self) or i >= len(self):
            raise IndexError
        return Box(
            self.xmin[i],
            self.xmax[i],
            self.ymin[i],
            self.ymax[i],
            self.img_w,
            self.img_h,
            self.label[i],
        )

    def check(self):
        assert isinstance(self.label, list), f"Require list, got {type(self.label)}."
        assert len(self.label) == len(self),\
            f"label length ({len(self.label)}) must equal len(self)({len(self)})."
        if len(self) > 0:
            assert all(isinstance(_l, type(self.label[0])) for _l in self.label),\
                "labels must be of same type."
        assert len(self.xmin) == len(self.xmax) == len(self.ymin) == len(self.ymax),\
            f"xmin({self.xmin}), xmax({len(self.xmax)}), ymin({self.ymin}),"\
            f"ymax({len(self.ymax)}) should be of same length."
        assert np.all(self.xmax > self.xmin),\
            f"xmax({self.xmax}) should be greater than xmin({self.xmin})."
        assert np.all(self.ymax > self.ymin),\
            f"ymax({self.ymax}) should be greater than ymin({self.ymin})."

    def is_within_image(self) -> np.ndarray:
        """
            Return a bool array whether each box is within the image.
        """
        return (self.xmin >= 0) & (self.xmax < self.img_w) & (self.ymin >= 0) & (self.ymax < self.img_h)

    def is_outside_image(self):
        """
            Return a bool array whether each box is completely outside the image.
        """
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
    def from_corner_points(cls, top_left_points, bottom_right_points, img_w: int = None, img_h: int = None):
        """
            create a box from top-left and bottom-right points
        """
        assert isinstance(top_left_points, PointArray),\
            f"Require PointArray, got {type(top_left_points)}."
        assert isinstance(bottom_right_points, PointArray),\
            f"Require PointArray, got {type(bottom_right_points)}."
        assert len(top_left_points) == len(bottom_right_points)
        assert len(top_left_points) > 0 or (img_w is not None and img_h is not None),\
            "img_w and img_h cannot be infered from empty list."
        img_w = top_left_points[0].img_w if img_w is None else img_w
        img_h = top_left_points[0].img_h if img_h is None else img_h

        return cls(
            xmin=top_left_points.x,
            ymin=top_left_points.y,
            xmax=bottom_right_points.x,
            ymax=bottom_right_points.y,
            img_w=img_w,
            img_h=img_h,
            label=top_left_points.label,
        )

    def to_points(self):
        """
            top-left, top-right, bottom-left, bottom-right points
        """
        return [
            PointArray(
                self.xmin,
                self.ymin,
                self.img_w,
                self.img_h,
                label=self.label,
            ),
            PointArray(
                self.xmax,
                self.ymin,
                self.img_w,
                self.img_h,
                label=self.label,
            ),
            PointArray(
                self.xmin,
                self.ymax,
                self.img_w,
                self.img_h,
                label=self.label,
            ),
            PointArray(
                self.xmax,
                self.ymax,
                self.img_w,
                self.img_h,
                label=self.label,
            ),
        ]

    @classmethod
    def from_boxes(cls, boxes: Iterable[Box], img_w: int = None, img_h: int = None):
        boxes = list(boxes)
        assert len(boxes) > 0 or (img_w is not None and img_h is not None),\
            "img_w and img_h cannot be infered from empty list."
        img_w = boxes[0].img_w if img_w is None else img_w
        img_h = boxes[0].img_h if img_h is None else img_h
        assert all([b.img_w == img_w for b in boxes]),\
            "Unequal img_w in boxes."
        assert all([b.img_h == img_h for b in boxes]),\
            "Unequal img_h in boxes."
        return cls(
            xmin=[b.xmin for b in boxes],
            xmax=[b.xmax for b in boxes],
            ymin=[b.ymin for b in boxes],
            ymax=[b.ymax for b in boxes],
            img_w=img_w,
            img_h=img_h,
            label=[b.label for b in boxes],
        )

    def to_boxes(self) -> List[Box]:
        return [
            Box(
                xmin=self.xmin[i],
                xmax=self.xmax[i],
                ymin=self.ymin[i],
                ymax=self.ymax[i],
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label[i],
            )
            for i in range(len(self))
        ]

    def to_oriented_box_array(self):
        """
           Return OrientedBoxArray are oriented to 1.5 pi (up-side).
        """
        from .oriented_box_array import OrientedBoxArray
        return OrientedBoxArray(
            x=self.center_x,
            y=self.center_y,
            w=self.width,
            h=self.height,
            theta=np.full(len(self), ONE_AND_HALF_PI, dtype=float),
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    @classmethod
    def from_numpy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        """
            array must be of shape (N, 4), representing xmin, xmax, ymin, ymax
        """
        assert array.ndim == 2 and array.shape[1] == 4,\
            f"array must be a np.ndarray shape=(N, 4), got {array.shape}."
        return cls(array[:, 0], array[:, 1], array[:, 2], array[:, 3], img_w, img_h, label)

    def to_numpy(self):
        """
            inverse of from_numpy()
        """
        return np.stack((self.xmin, self.xmax, self.ymin, self.ymax), 1)

    @classmethod
    def from_numpy_as_xyxy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.ndim == 2 and array.shape[1] == 4,\
            f"array must be a np.ndarray shape=(N, 4), got {array.shape}."
        xmin, ymin, xmax, ymax = array[:, 0], array[:, 1], array[:, 2], array[:, 3]
        return cls(xmin, xmax, ymin, ymax, img_w, img_h, label)

    def to_numpy_as_xyxy(self):
        return np.stack([self.xmin, self.ymin, self.xmax, self.ymax], 1)

    @classmethod
    def from_numpy_as_cxcywh(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.ndim == 2 and array.shape[1] == 4,\
            f"array must be a np.ndarray shape=(N, 4), got {array.shape}."
        cx, cy, w, h = array[:, 0], array[:, 1], array[:, 2], array[:, 3]
        half_w, half_h = w/2, h/2
        xmin, xmax, ymin, ymax = cx - half_w, cx + half_w, cy - half_h, cy + half_h
        return cls(xmin, xmax, ymin, ymax, img_w, img_h, label)

    def to_numpy_as_cxcywh(self):
        return np.stack([self.center_x, self.center_y, self.width, self.height], 1)

    @classmethod
    def from_numpy_as_xywh(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.ndim == 2 and array.shape[1] == 4,\
            f"array must be a np.ndarray shape=(N, 4), got {array.shape}."
        xmin, ymin, w, h = array[:, 0], array[:, 1], array[:, 2], array[:, 3]
        xmax, ymax = xmin + w, ymin + h
        return cls(xmin, xmax, ymin, ymax, img_w, img_h, label)

    def to_numpy_as_xywh(self):
        return np.stack([self.xmin, self.ymin, self.width, self.height], 1)

    def select(self, indices):
        """
        Select boxes by arbitrary indexing.

        Args:
            indices (Any): Any object that is legal for 1-D np.ndarray indexing

        Returns:
            BoxArray
        """
        return BoxArray(
            xmin=self.xmin[indices],
            xmax=self.xmax[indices],
            ymin=self.ymin[indices],
            ymax=self.ymax[indices],
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
            BoxArray: duplicate removed box array
        """
        iou_matrix = self.iou(self)
        duplicate_matrix = iou_matrix >= iou_threshold
        preserved = nms(duplicate_matrix)
        return self.select(np.nonzero(preserved)[0])

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
        """
            width/height, as a np.ndarray of shape (len(self), )
        """
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
        s = f"BoxArray(img_w={self.img_w}, img_h={self.img_h}, xmin=..., xmax=..., ymin=..., ymax=..., label):\n"
        for b in self:
            s += f"({b.xmin}, {b.xmax}, {b.ymin}, {b.ymax}), {b.label},\n"
        return s

    # transforms

    def _clip(self):
        is_within = self.is_within_image()
        is_outside = self.is_outside_image()
        if np.all(is_within) or np.all(is_outside):
            return self
        else:
            is_kept = ~is_outside
            return BoxArray(
                xmin=np.maximum(self.xmin[is_kept], 0),
                xmax=np.minimum(self.xmax[is_kept], self.img_w),
                ymin=np.maximum(self.ymin[is_kept], 0),
                ymax=np.minimum(self.ymax[is_kept], self.img_h),
                img_w=self.img_w,
                img_h=self.img_h,
                label=[self.label[i] for i, keep in enumerate(is_kept) if keep],
            )

    def _pad(self, up, down, left, right, fill_value):
        timage = self._base_image.pad(up, down, left, right, fill_value)
        return BoxArray(
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
        return BoxArray(
            xmin=self.xmin - xmin,
            xmax=self.xmax - xmin,
            ymin=self.ymin - ymin,
            ymax=self.ymax - ymin,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    def _horizontal_flip(self):
        return BoxArray(
            xmin=self.img_w - self.xmax,
            xmax=self.img_w - self.xmin,
            ymin=self.ymin,
            ymax=self.ymax,
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    def _vertical_flip(self):
        return BoxArray(
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.img_h - self.ymax,
            ymax=self.img_h - self.ymin,
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    def _rotate(self, angle):
        """
        Rotate the box with the image by some degree. This function returns an OrientedBox.
        """
        return self.to_oriented_box_array().rotate(angle)

    def _rotate_right_angle(self, rotate_right_angle):
        timage = self._base_image.rotate_right_angle(rotate_right_angle)
        if rotate_right_angle == 90:
            return BoxArray(
                xmin=self.img_h - self.ymax,
                xmax=self.img_h - self.ymin,
                ymin=self.xmin,
                ymax=self.xmax,
                img_w=timage.img_w,
                img_h=timage.img_h,
                label=self.label,
            )
        elif rotate_right_angle == 180:
            return BoxArray(
                xmin=self.img_w - self.xmax,
                xmax=self.img_w - self.xmin,
                ymin=self.img_h - self.ymax,
                ymax=self.img_h - self.ymin,
                img_w=timage.img_w,
                img_h=timage.img_h,
                label=self.label,
            )
        elif rotate_right_angle == 270:
            return BoxArray(
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
        return BoxArray(
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
        return BoxArray(
            xmin=self.ymin,
            xmax=self.ymax,
            ymin=self.xmin,
            ymax=self.xmax,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label,
        )

    # relations

    def intersection_area(self, other) -> np.ndarray:
        """
            Compute the intersection area between each pair of boxes.

        Args:
            other (BoxArray): another BoxArray

        Returns:
            np.ndarray: shape (len(self), len(other)) int ndarray
        """
        assert isinstance(other, BoxArray),\
            "Currently I/U/IoU computation is only supported for Box-Box or BoxArray-BoxArray."
        assert self.img_w == other.img_w and self.img_h == other.img_h,\
            f"Boxes must have save base image shape self({self.img_w}, {self.img_h}) v.s."\
            f"box({other.img_w}, {other.img_h})."
        inter_xmin = np.maximum.outer(self.xmin, other.xmin)
        inter_xmax = np.minimum.outer(self.xmax, other.xmax)
        inter_ymin = np.maximum.outer(self.ymin, other.ymin)
        inter_ymax = np.minimum.outer(self.ymax, other.ymax)

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        zero_mask = (inter_xmin > inter_xmax) | (inter_ymin > inter_ymax)
        inter_area[zero_mask] = 0
        return inter_area

    def union_area(self, other) -> np.ndarray:
        """
            Compute the union area between each pair of boxes.

        Args:
            other (BoxArray): another BoxArray

        Returns:
            np.ndarray: shape (len(self), len(other)) int ndarray
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
        if not isinstance(boxarr, BoxArray):
            raise TypeError()
        return self.img_w == boxarr.img_w and\
            self.img_h == boxarr.img_h and\
            np.all(self.xmin == boxarr.xmin) and\
            np.all(self.xmax == boxarr.xmax) and\
            np.all(self.ymin == boxarr.ymin) and\
            np.all(self.ymax == boxarr.ymax) and\
            self.label == boxarr.label

    def __add__(self, boxarr_or_box):
        assert boxarr_or_box.img_w == self.img_w and boxarr_or_box.img_h == self.img_h
        if isinstance(boxarr_or_box, Box):
            return BoxArray(
                xmin=np.concatenate([self.xmin, [boxarr_or_box.xmin]]),
                xmax=np.concatenate([self.xmax, [boxarr_or_box.xmax]]),
                ymin=np.concatenate([self.ymin, [boxarr_or_box.ymin]]),
                ymax=np.concatenate([self.ymax, [boxarr_or_box.ymax]]),
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label + [boxarr_or_box.label],
            )
        elif isinstance(boxarr_or_box, BoxArray):
            return BoxArray(
                xmin=np.concatenate([self.xmin, boxarr_or_box.xmin]),
                xmax=np.concatenate([self.xmax, boxarr_or_box.xmax]),
                ymin=np.concatenate([self.ymin, boxarr_or_box.ymin]),
                ymax=np.concatenate([self.ymax, boxarr_or_box.ymax]),
                img_w=self.img_w,
                img_h=self.img_h,
                label=self.label + boxarr_or_box.label,
            )
        else:
            raise TypeError()
