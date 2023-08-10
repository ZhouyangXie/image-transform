from typing import List, Hashable

import numpy as np
import cv2

from .basic import ImageAnnotation
from .image import EmptyImage
from .utils import IsWithinImage, rle2sequence, sequence2rle, pad_image, rotate_image
from .label import Label, Empty, Scoped, MultipleScoped, ScopedMaskWithConfidence, ArbitraryHashable


class Mask(ImageAnnotation):
    def __init__(self, mask: np.ndarray, label: Label = None) -> None:
        """
        Args:
            mask (np.ndarray): label mask. Label 0 is the background.
                dtype is one of np.uint8 or np.int32.
            label List[Hashable]: class label of each value(>=1) in the mask.
        """
        super().__init__(label)
        if isinstance(self.label, list) and all([isinstance(_l, ArbitraryHashable) for _l in self.label]):
            values = list(set(_l.value for _l in self.label))
            self.label = type("_TempMultipleScopedForMask", (MultipleScoped, ), {"scope": values})([])
        self._base_image = EmptyImage(mask.shape[1], mask.shape[0])
        self.mask = mask
        self.check()

    def check(self):
        assert isinstance(self.label, (Empty, Scoped, MultipleScoped, ScopedMaskWithConfidence)),\
            f"self.label must be Empty, Scoped, MultipleScoped or ScopedMaskWithConfidence. Got {type(self.label)}."
        assert (len(self.scope) >= np.max(self.mask)),\
            f"Scope size ({len(self.scope)}) must NOT be less than mask.max()({np.max(self.mask)})"
        assert self.mask.shape == (self.img_h, self.img_w,),\
            f"mask shape ({self.mask.shape}) should be equal to image (H{self.img_h}, W{self.img_w})"
        assert (self.mask.dtype in (np.uint8, np.int32)),\
            f"data type ({self.mask.dtype}) should be one of np.uint8 or np.int32."
        if isinstance(self.label, ScopedMaskWithConfidence):
            assert self.mask.shape == self.label.confidence.shape,\
                f"Mismatched shape for mask {self.mask.shape} and confidence {self.label.confidence.shape}"

    def check_within_image(self):
        return IsWithinImage.YES

    def copy(self):
        """
            If we do not override `copy` or `__deepcopy__`, `__getstate__` will be called and comparess the mask,
            which is not desired for deep copy.
        """
        return Mask(self.mask.copy(), self.label)

    # conversions
    def to_polygon(self):
        # TODO: cv2.findContour might be appropriate
        raise NotImplementedError("still looking for a fast and accurate implementation")

    @classmethod
    def from_polygons(cls, polygons, img_w: int = None, img_h: int = None):
        """
        Polygons should have the same type Empty or Scoped label.
        Pixels in overlapping areas of polygons will be assigned to label of max index.
        labelled by the label of the later polygon.
        """
        polygons = list(polygons)
        assert len(polygons) > 0 or (img_w is not None and img_h is not None),\
            "img_w and img_h cannot be infered from empty list."
        img_w = polygons[0].img_w if img_w is None else img_w
        img_h = polygons[0].img_h if img_h is None else img_h

        assert all(p.img_w == polygons[0].img_w for p in polygons), "Polygons must have same img_w."
        assert all(p.img_h == polygons[0].img_h for p in polygons), "Polygons must have same img_h."
        assert all(isinstance(p.label, type(polygons[0].label)) for p in polygons),\
            "Polygons must have same label type."

        masks = [polygon.to_mask().mask for polygon in polygons]

        if len(masks) == 0:
            merged_mask = np.zeros(img_h, img_w)
            label = None
        elif len(masks) == 1:
            merged_mask = masks[0]
            label = polygons[0].label
        else:
            merged_mask = np.maximum(*masks)
            label = polygons[0].label

        return Mask(merged_mask, label)

    @classmethod
    def from_numpy(cls, array: np.ndarray, label: Label = None):
        return cls(array, label)

    def to_numpy(self):
        return self.mask

    # properties
    @property
    def scope(self) -> List[Hashable]:
        """
            Semantic meaning of the value in self.mask. 0 always stands for background.
            Value 1, 2, 3... respectively stands for the 0-th, 1-st, 2nd... label in the scope.
            This scope is the same as Label.scope if self.label is *Scoped*.
        """
        if isinstance(self.label, (Scoped, MultipleScoped, ScopedMaskWithConfidence)):
            return self.label.scope
        else:
            # Empty
            return [True, ]

    @property
    def unique_labels(self):
        """
            A subset of self.scope. Only those exist in self.mask will be returned.
        """
        unique_values = np.unique(self.mask)
        return [self.scope[int(uv) - 1] for uv in unique_values if uv != 0]

    @property
    def img_w(self):
        return self._base_image.width

    @property
    def img_h(self):
        return self._base_image.height

    def __repr__(self):
        return (f"Mask(img_w={self.img_w}, img_h={self.img_h}, num_label={len(self.scope)}, label={self.label}")

    # transformations

    def _clip(self):
        return self

    def _pad(self, up, down, left, right, fill_value=None):
        return Mask(
            pad_image(self.mask, up, down, left, right, 0),
            label=self.label._pad(up, down, left, right)
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        return Mask(
            self.mask[ymin:ymax + 1, xmin:xmax + 1],
            label=self.label._crop(xmin, xmax, ymin, ymax)
        )

    def _horizontal_flip(self):
        return Mask(self.mask[:, ::-1], label=self.label._horizontal_flip())

    def _vertical_flip(self):
        return Mask(self.mask[::-1, :], label=self.label._vertical_flip())

    def _rotate(self, angle):
        mask = self.mask.astype(np.uint16) if self.mask.dtype == np.int32 else self.mask
        return Mask(rotate_image(mask, angle).astype(self.mask.dtype), label=self.label._rotate(angle))

    def _rotate_right_angle(self, rotate_right_angle):
        label = self.label._rotate_right_angle(rotate_right_angle)
        if rotate_right_angle == 90:
            return Mask(self.mask.transpose((1, 0))[:, ::-1], label=label)
        elif rotate_right_angle == 180:
            return Mask(self.mask[::-1, ::-1], label=label)
        elif rotate_right_angle == 270:
            return Mask(self.mask.transpose((1, 0))[::-1, :], label=label)
        else:
            return Mask(self.mask, label=label)

    def _resize(self, dst_w, dst_h):
        mask = cv2.resize(self.mask, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST).astype(self.mask.dtype)
        return Mask(mask, label=self.label._resize(dst_w, dst_h))

    def _transpose(self):
        return Mask(self.mask.transpose((1, 0)), label=self.label._transpose())

    def __eq__(self, mask):
        if not isinstance(mask, Mask):
            raise TypeError()
        return self.scope == mask.scope and np.all(self.mask == mask.mask)

    # for pickling
    def __getstate__(self):
        rle_len, rle_val, dtype = sequence2rle(self.mask)
        if rle_len.nbytes + rle_val.nbytes < self.mask.nbytes:
            mask = (rle_len, rle_val, dtype)
        else:
            mask = self.mask
        return {
            "_base_image": self._base_image,
            "label": self.label,
            "mask": mask,
        }

    def __setstate__(self, state):
        assert isinstance(state, dict)
        self._base_image = state["_base_image"]
        self.label = state["label"]
        mask = state["mask"]
        if isinstance(mask, tuple):
            mask = rle2sequence(mask).reshape((self._base_image.img_w, self._base_image.img_h))
        self.mask = mask
