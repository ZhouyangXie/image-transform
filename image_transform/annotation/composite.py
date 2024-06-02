from typing import Iterable, Union
from collections.abc import MutableSequence
from itertools import chain
from functools import reduce

from .basic import ImageAnnotation
from .image import Image, EmptyImage
from .point import Point
from .point_array import PointArray
from .box import Box
from .box_array import BoxArray
from .oriented_box import OrientedBox
from .oriented_box_array import OrientedBoxArray
from .mask import Mask
from .label import Label, Empty, Scoped, MultipleScoped, ScopedWithConfidence, ScopedMaskWithConfidence,\
    ArbitraryHashable, ProbabilisticMultipleScoped
from .utils import IsWithinImage


class Composite(ImageAnnotation, MutableSequence):
    def __init__(
            self, annotations: Iterable[ImageAnnotation],
            img_w: int = None, img_h: int = None
            ):
        """
        Args:
            annotations (Iterable[ImageAnnotation]): annotation and images. Not empty.
            img_w (int, optional):  Defaults to None.
            img_h (int, optional):  Defaults to None.
        """
        super().__init__()
        annotations = list(annotations)
        if len(annotations) == 0:
            assert img_w is not None and img_h is not None,\
                "when annotations are None by default, img_w/img_h must be specified"
        self._img_w = annotations[0].img_w if img_w is None else img_w
        self._img_h = annotations[0].img_h if img_h is None else img_h
        self.annotations = annotations
        self.check()

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def check_annotation(self, a):
        assert a.img_w == self.img_w and a.img_h == self.img_h,\
            f"annotation has base image size({a.img_w}, {a.img_h}), different from"\
            f" object base image size({self.img_w}, {self.img_h})"

    def check(self):
        for i, a in enumerate(self.annotations):
            self.check_annotation(a)
            a.check()

    def check_within_image(self):
        is_within_image = [a.check_within_image() for a in self.annotations]
        if all([i == IsWithinImage.YES for i in is_within_image]):
            return IsWithinImage.YES
        elif all([i == IsWithinImage.NO for i in is_within_image]):
            return IsWithinImage.NO
        else:
            return IsWithinImage.PARTIAL

    @classmethod
    def from_numpy(cls):
        raise TypeError('Not supported. Refer to transforms.ToNumpy instead.')

    def to_numpy(self):
        return [a.to_numpy() for a in self.annotations]

    def __repr__(self):
        s = "Composite:[\n"
        for a in self.annotations:
            s += "\t" + str(a) + "\n"
        s += "]\n"
        return s

    @property
    def img_w(self) -> int:
        return self._img_w

    @property
    def img_h(self) -> int:
        return self._img_h

    @property
    def images(self):
        return [a for a in self.annotations if isinstance(a, (Image, EmptyImage))]

    @property
    def image(self):
        images = self.images
        if len(images) > 0:
            return images[0]
        else:
            return None

    @property
    def num_image(self) -> int:
        return len(self.images)

    @property
    def boxes(self) -> BoxArray:
        boxes = [anno for anno in self.annotations if isinstance(anno, Box)]
        box_arrs = [anno for anno in self.annotations if isinstance(anno, BoxArray)]
        box_arrs.append(BoxArray.from_boxes(boxes, self.img_w, self.img_h))
        box_arr = reduce(lambda a, b: a + b, box_arrs)
        return box_arr

    @property
    def points(self) -> PointArray:
        points = [anno for anno in self.annotations if isinstance(anno, Point)]
        point_arrs = [anno for anno in self.annotations if isinstance(anno, PointArray)]
        point_arrs.append(PointArray.from_points(points, self.img_w, self.img_h))
        point_arr = reduce(lambda a, b: a + b, point_arrs)
        return point_arr

    @property
    def oriented_boxes(self) -> OrientedBoxArray:
        oriented_boxes = [anno for anno in self.annotations if isinstance(anno, OrientedBox)]
        oriented_box_arrs = [anno for anno in self.annotations if isinstance(anno, OrientedBoxArray)]
        oriented_box_arrs.append(OrientedBoxArray.from_oriented_boxes(oriented_boxes, self.img_w, self.img_h))
        oriented_box_arr = reduce(lambda a, b: a + b, oriented_box_arrs)
        return oriented_box_arr

    @property
    def unique_labels(self):
        def _get_unique_labels(a: Union[Label, ImageAnnotation]) -> set:
            if isinstance(a, Mask):
                return set(a.scope)
            if isinstance(a, ImageAnnotation):
                label = a.label
            else:
                assert isinstance(a, (Label, list))
                label = a
            if isinstance(label, Empty):
                return {None}
            elif isinstance(label, (Scoped, ArbitraryHashable, ScopedWithConfidence)):
                return {label.value}
            elif isinstance(label, MultipleScoped):
                return set(label.values)
            elif isinstance(label, (ProbabilisticMultipleScoped, ScopedMaskWithConfidence)):
                return set(label.scope)
            elif isinstance(label, list):
                return set(chain(*[_get_unique_labels(_label) for _label in label]))
            else:
                return {None}
        return set(chain(*[_get_unique_labels(a) for a in self.annotations]))

    def _clip(self):
        return self.new([a.clip() for a in self.annotations])

    def _pad(self, up, down, left, right, fill_value):
        return self.new([a.pad(up, down, left, right, fill_value) for a in self.annotations])

    def _crop(self, xmin, xmax, ymin, ymax):
        return self.new([a.crop(xmin, xmax, ymin, ymax) for a in self.annotations])

    def _horizontal_flip(self):
        return self.new([a.horizontal_flip() for a in self.annotations])

    def _vertical_flip(self):
        return self.new([a.vertical_flip() for a in self.annotations])

    def _rotate(self, angle):
        return self.new([a.rotate(angle) for a in self.annotations])

    def _rotate_right_angle(self, angle):
        return self.new([a.rotate_right_angle(angle) for a in self.annotations])

    def _rescale(self, factor_x, factor_y):
        return self.new([a.rescale(factor_x, factor_y) for a in self.annotations])

    def _resize(self, target_w, target_h):
        return self.new([a.resize(target_w, target_h) for a in self.annotations])

    def _transpose(self):
        return self.new([a.transpose() for a in self.annotations])

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    def __getitem__(self, i):
        ret = self.annotations[i]
        if isinstance(ret, list):
            return self.new(ret)
        else:
            return ret

    def __delitem__(self, i):
        del self.annotations[i]

    def __setitem__(self, i, a):
        self.check_annotation(a)
        self.annotations[i] = a

    def insert(self, i, a):
        self.check_annotation(a)
        self.annotations.insert(i, a)

    def select(self, indices):
        return self.new([self.annotations[i] for i in indices], self.img_w, self.img_h)

    def filter_within_image(self, is_within_image: IsWithinImage = None):
        """
            Filter the image annotations by whether it is inside the image.

        Args:
            is_within_image (IsWithinImage, optional):
                Remove a shape if it is not completely or partly within the image.
                If IsWithinImage.NO or None (default), filtering is not performed.
                If IsWithinImage.PARTIAL, annotations completely outside the image are removed.
                If IsWithinImage.YES, only annotations completely inside the image are kept.
        """
        if is_within_image in (IsWithinImage.NO, None):
            return self
        else:
            assert is_within_image in (IsWithinImage.PARTIAL, IsWithinImage.YES),\
                "is_within_image must be of type IsWithinImage"

        return self.select([
            i for i, a in enumerate(self.annotations)
            if a.check_within_image() in (IsWithinImage.YES, is_within_image)
        ])

    def filter_type(self, types: Iterable[ImageAnnotation] = None):
        if types is None:
            return self
        else:
            types = tuple(types)
            return self.select([
                i for i, a in enumerate(self.annotations) if isinstance(a, types)
            ])

    def __eq__(self, composite):
        raise TypeError('Composite comparison is not supported.')

    def flatten(self, points=True, boxes=True, oriented_boxes=True):
        """
            for each of X in self.annotations:
                PointArray -> List[Point], if points;
                BoxArray -> List[Box], if boxes;
                OrientedBoxArray -> List[OrientedBox], if oriented_boxes;
        """
        annotations = []
        for anno in self.annotations:
            if isinstance(anno, PointArray) and points:
                annotations.extend(anno.to_points())
            elif isinstance(anno, BoxArray) and boxes:
                annotations.extend(anno.to_boxes())
            elif isinstance(anno, OrientedBoxArray) and oriented_boxes:
                annotations.extend(anno.to_oriented_boxes())
            else:
                annotations.append(anno)

        return self.new(annotations, self.img_w, self.img_h)

    def compact(self, points=True, boxes=True, oriented_boxes=True):
        """
            for each of X in self.annotations:
                List[Point]/PointArray -> PointArray, if points;
                List[Box]/BoxArray -> BoxArray , if boxes;
                List[OrientedBox]/OrientedBoxArray -> OrientedBoxArray, if oriented_boxes;
        """
        annotations = []
        for anno in self.annotations:
            if isinstance(anno, (PointArray, Point)) and points:
                continue
            elif isinstance(anno, (BoxArray, Box)) and boxes:
                continue
            elif isinstance(anno, (OrientedBoxArray, OrientedBox)) and oriented_boxes:
                continue
            else:
                annotations.append(anno)

        if points:
            annotations.append(self.points)
        if boxes:
            annotations.append(self.boxes)
        if oriented_boxes:
            annotations.append(self.oriented_boxes)

        return self.new(annotations, self.img_w, self.img_h)
