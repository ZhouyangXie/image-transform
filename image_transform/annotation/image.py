from __future__ import annotations
import cv2
import numpy as np
from typing import Type

from .basic import ImageAnnotation
from .label import Label
from .utils import get_rotated_image_size, IsWithinImage, rotate_image, pad_image


ALLOWED_DATA_TYPE = (np.uint8, np.float32, np.float64)


class EmptyImage(ImageAnnotation):
    def __init__(
            self, width: int, height: int, channels: int = 3, dtype: Type = np.uint8, label: Label = None
            ) -> None:
        super().__init__(label)
        assert (width > 0 and height > 0 and channels > 0),\
            f"width:({width}) height:({height}) channels:({channels}) must all be positive"
        assert (dtype in ALLOWED_DATA_TYPE),\
            f"Image/EmptyImage dtype should be one of {ALLOWED_DATA_TYPE}"
        self.width, self.height = int(width), int(height)
        self.channels = int(channels)
        self.dtype = dtype

    @property
    def data(self) -> np.ndarray:
        return np.empty((self.height, self.width, self.channels), dtype=self.dtype)

    def to_image(self):
        return Image(np.zeros((self.height, self.width, self.channels), dtype=self.dtype), label=self.label)

    @staticmethod
    def from_numpy(cls, ndarray: np.ndarray):
        raise NotImplementedError(f"{cls.__name__}.from_numpy() should not be called")

    def to_numpy(self):
        """
            return np.empty((img_h, img_w, channels), self.dtype)
        """
        return self.data

    def check_within_image(self):
        return IsWithinImage.YES

    # properties

    @property
    def img_w(self) -> int:
        return self.width

    @property
    def img_h(self) -> int:
        return self.height

    @property
    def depth(self) -> int:
        return self.channels

    @property
    def data_type(self):
        return self.dtype

    def __repr__(self):
        return f"Image img_w={self.img_w}, img_h={self.img_h}, channels={self.channels},"\
            f"dtype={self.dtype}, label={self.label}"

    # transformations
    def _clip(self):
        return self

    def _pad(self, up, down, left, right, fill_value=None):
        return EmptyImage(
            width=self.width + left + right,
            height=self.height + up + down,
            channels=self.channels,
            dtype=self.dtype,
            label=self.label,
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        return EmptyImage(
            width=xmax - xmin + 1,
            height=ymax - ymin + 1,
            channels=self.channels,
            dtype=self.dtype,
            label=self.label,
        )

    def _horizontal_flip(self):
        return self

    def _vertical_flip(self):
        return self

    def _rotate(self, angle):
        rw, rh = get_rotated_image_size(self.width, self.height, angle)
        return EmptyImage(
            width=rw,
            height=rh,
            channels=self.channels,
            dtype=self.dtype,
            label=self.label
        )

    def _resize(self, dst_w, dst_h):
        return EmptyImage(
            width=dst_w,
            height=dst_h,
            channels=self.channels,
            dtype=self.dtype,
            label=self.label
        )

    def _transpose(self):
        return EmptyImage(
            width=self.height,
            height=self.width,
            channels=self.channels,
            dtype=self.dtype,
            label=self.label
        )

    def __eq__(self, image):
        if not isinstance(image, EmptyImage):
            raise TypeError()
        return self.img_w == image.img_w and\
            self.img_h == image.img_h and\
            self.channels == image.channels and\
            self.dtype == image.dtype and\
            self.label == image.label


class Image(ImageAnnotation):
    """
    Numpy based (H, W, C) shape image data
    """

    def __init__(self, data: np.ndarray, label: Label = None) -> None:
        """
        Args:
            data (np.ndarray): image data, should be a 3-dim (H, W, C) or 2-dim (H, W) ndarray
                data type in image.ALLOWED_DATA_TYPE. Note that the number of the color channel
                restricts some color-involved image transformations.
        """
        super().__init__(label)
        if data.ndim == 2:
            data = data.reshape((*data.shape, 1))

        self.data = data
        self.check()

    def check(self):
        assert self.data.ndim == 3, f"data dim {self.data.ndim} is invalid, should be 3"
        assert (self.data.dtype in ALLOWED_DATA_TYPE),\
            f"data type {self.data.dtype} is not one of the supported: {ALLOWED_DATA_TYPE}"

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray, label: Label = None):
        """
            Same as Image(array, label)
        """
        return cls(ndarray, label)

    def to_numpy(self) -> np.ndarray:
        """
            return image data shape=(img_h, img_w, channels) dtype=self.dtype
        """
        return self.data

    # conversions

    def copy(self):
        return Image(self.data.copy(), label=self.label)

    @classmethod
    def from_pil(cls, image, label: Label = None):
        return Image(np.array(image), label=label)

    def to_pil(self):
        import PIL.Image
        data = self.data if self.channels > 1 else self.data[..., 0]
        return PIL.Image.fromarray(data)

    def to_empty_image(self):
        return EmptyImage(
            width=self.width,
            height=self.height,
            channels=self.channels,
            dtype=self.data.dtype,
            label=self.label,
        )

    def check_within_image(self):
        return IsWithinImage.YES

    # properties

    @property
    def img_w(self) -> int:
        return self.width

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def img_h(self) -> int:
        return self.height

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def depth(self) -> int:
        return self.data.shape[2]

    @property
    def channels(self) -> int:
        return self.depth

    @property
    def data_type(self) -> Type:
        return self.data.dtype

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"Image img_w={self.img_w}, img_h={self.img_h}, channels={self.channels},"\
            f"dtype={self.dtype}, label={self.label}"

    # transformations

    def _clip(self):
        return self

    def _pad(self, up, down, left, right, fill_value):
        return Image(
            pad_image(self.data, up, down, left, right, fill_value),
            label=self.label,
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        crop = self.data[ymin:ymax + 1, xmin:xmax + 1, :]
        return Image(crop, label=self.label)

    def _horizontal_flip(self):
        return Image(self.data[:, ::-1, :], label=self.label)

    def _vertical_flip(self):
        return Image(self.data[::-1, :, :], label=self.label)

    def _rotate(self, angle):
        return Image(rotate_image(self.data, angle, cv2.INTER_LINEAR), label=self.label)

    def _rotate_right_angle(self, angle):
        angle = angle % 360
        if angle == 90:
            return Image(self.data.transpose((1, 0, 2))[:, ::-1, :], label=self.label)
        elif angle == 180:
            return Image(self.data[::-1, ::-1, :], label=self.label)
        elif angle == 270:
            return Image(self.data.transpose((1, 0, 2))[::-1, :, :], label=self.label)
        else:
            return Image(self.data, label=self.label)

    def _resize(self, dst_w, dst_h):
        return Image(cv2.resize(self.data, (dst_w, dst_h)), label=self.label)

    def _transpose(self):
        return Image(self.data.transpose((1, 0, 2)), label=self.label)

    def __eq__(self, image):
        if not isinstance(image, Image):
            raise TypeError()
        return self.data_type == image.data_type and\
            np.all(self.data == image.data) and\
            self.label == image.label
