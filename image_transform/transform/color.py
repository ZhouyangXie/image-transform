from abc import abstractmethod
from typing import Iterable

import numpy as np
import cv2

from ..annotation import Image, EmptyImage, Composite
from .basic import Transform, EmptyTransform


class ColorTransform(Transform):
    """
        Abstract class for transformations that change pixel values
        but do not change image size or data type. A subclass needs
        to have _color_transform implemented.

        Image/EmptyImage/Composite objects are accepted to be transformed.
        Other types are not processed.

        _get_inverse can be overriden if the inverse transformation is possible.

    """
    @abstractmethod
    def _color_transform(self, data):
        return data

    def _transform(self, image):
        if isinstance(image, Image):
            return Image(self._color_transform(image.data), image.label)
        elif isinstance(image, EmptyImage):
            data = self._color_transform(image.data)
            return EmptyImage(data.shape[1], data.shape[0], data.shape[2], data.dtype, image.label)
        elif isinstance(image, Composite):
            return Composite([self._transform(a) for a in image.annotations], image.img_w, image.img_h)
        else:
            return image

    def _get_inverse(self):
        return EmptyTransform()


class Normalize(ColorTransform):
    """
    0-mean, 1-std normalization.

    Args:
        mean (Iterable[dtype]): length should equal image channels,
            and dtype should be same as image dtype.
        std (Iterable[dtype]): same as mean.
        eps (float, optional): small factor to prevent zero std. Defaults to 1e-1000.
    """
    def __init__(self, mean: Iterable, std: Iterable, eps=1e-1000):
        super().__init__()
        self.mean = np.array(mean)
        if isinstance(self.mean, np.ndarray):
            assert self.mean.ndim == 1
            self.channels = self.mean.shape[0]
        else:
            assert np.isscalar(self.mean)
            self.channels = 1

        self.std = np.maximum(np.array(std), eps)
        if isinstance(self.std, np.ndarray):
            assert self.std.shape == (self.channels, )
        else:
            assert np.isscalar(self.std) and self.channels == 1

        if self.mean.dtype not in (np.float32, np.float64):
            self.mean = self.mean.astype(np.float32)
        if self.std.dtype not in (np.float32, np.float64):
            self.std = self.std.astype(np.float32)

    def _color_transform(self, data):
        assert data.dtype in (np.float32, np.float64)
        channels = data.shape[-1]
        assert channels == self.channels,\
            f"Normalize is initialized for channels={self.channels}, got image channels={channels}"
        return ((data - self.mean)/self.std).astype(data.dtype)

    def _get_inverse(self):
        re_std = 1/self.std
        return Normalize(-self.mean * re_std, re_std)


class GaussianBlur(ColorTransform):
    """
    Gaussian blurring.

    Args:
        kernel_size_x (int): Gaussian kernel size on the x-axis.
        kernel_size_y (int, optional): Gaussian kernel size on the y-axis.
            If None, same as kernel_size_x. Defaults to None.
        sigma_x (float, optional): Gaussian kernel sigma on the x-axis. Defaults to 1.0.
        sigma_y (float, optional): Gaussian kernel sigma on the Y-axis.
            If None, same as sigma_x. Defaults to None.
    """
    def __init__(
            self, kernel_size_x: int, kernel_size_y: int = None,
            sigma_x: float = 1.0, sigma_y: float = None):
        super().__init__()
        assert isinstance(kernel_size_x, int) and kernel_size_x >= 1
        kernel_size_y = kernel_size_x if kernel_size_y is None else kernel_size_y
        assert isinstance(kernel_size_y, int) and kernel_size_y >= 1
        assert isinstance(sigma_x, float) and sigma_x > 0
        sigma_y = sigma_x if sigma_y is None else sigma_y
        assert isinstance(sigma_y, float) and sigma_y > 0
        self.kernel_size = (kernel_size_x, kernel_size_y)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def _color_transform(self, data):
        return cv2.GaussianBlur(data, self.kernel_size, self.sigma_x, self.sigma_y)


class GaussianNoise(ColorTransform):
    """
        Add N(0, sigma) noise to the image data.
    """
    def __init__(self, sigma=1):
        assert sigma > 0, f"sigma must be positive, got {sigma}"
        self.sigma = sigma

    def _color_transform(self, data):
        dtype = data.dtype
        noise = np.random.normal(loc=0., scale=self.sigma, size=data.shape)
        data = noise + data
        if dtype == np.uint8:
            return data.clip(0, 255).astype(dtype)
        else:
            return data.clip(0, 1).astype(dtype)


class ToDataType(ColorTransform):
    def __init__(self, to_dtype):
        super().__init__()
        assert to_dtype in (
            np.uint8, np.float32, np.float64, 'uint8', 'float32', 'float64', 'byte', 'fp32', 'fp64')
        if isinstance(to_dtype, str):
            to_dtype = {
                'uint8': np.uint8, 'float32': np.float32, 'float64': np.float64,
                'byte': np.uint8, 'fp32': np.float32, 'fp64': np.float64,
            }[to_dtype]
        self.to_dtype = to_dtype
        self.from_dtype = None

    def _color_transform(self, data):
        self.from_dtype = data.dtype
        if data.dtype == self.to_dtype:
            return data
        if data.dtype == np.uint8:
            # uint8 -> float32/float64
            return (data/255).astype(self.to_dtype)
        elif self.to_dtype == np.uint8:
            # float32/float64 -> uint8
            return (data * 255).astype(self.to_dtype)
        else:
            # float32 <-> float64
            return data.astype(self.to_dtype)

    def _get_inverse(self):
        return ToDataType(self.from_dtype)


class Gray2RGB(ColorTransform):
    def __init__(self):
        self.is_converted = None
        super().__init__()

    def _color_transform(self, data):
        c = data.shape[2]
        if c == 1:
            dtype = data.dtype
            if dtype == np.float64:
                data = data.astype(np.float32)
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB).astype(dtype)
            self.is_converted = True
        else:
            self.is_converted = False

        return data

    def _get_inverse(self):
        return RGB2Gray() if self.is_converted else EmptyTransform()


class RGB2Gray(ColorTransform):
    def __init__(self):
        self.is_converted = None
        super().__init__()

    def _color_transform(self, data):
        c = data.shape[2]
        if c == 3:
            dtype = data.dtype
            if dtype == np.float64:
                data = data.astype(np.float32)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY).astype(dtype)
            self.is_converted = True
        else:
            self.is_converted = False

        return data

    def _get_inverse(self):
        return Gray2RGB() if self.is_converted else EmptyTransform()
