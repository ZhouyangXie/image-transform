from abc import abstractmethod, abstractproperty, ABC, abstractclassmethod
from typing import Union, List, Hashable
from copy import deepcopy
from math import ceil

import numpy as np
from .utils import IsWithinImage, normalize
from .label import Label, Empty, ArbitraryHashable, is_hashable


class ImageAnnotation(ABC):
    def __init__(self, label: Union[Hashable, Label, List[Union[Label, Hashable]]] = None):
        if label is None:
            self.label = Empty()
        elif isinstance(label, Label):
            self.label = label.copy()
        elif is_hashable(label):
            self.label = ArbitraryHashable(label)
        else:
            assert isinstance(label, list), f"invalid label type {type(label)}"
            self.label = []
            for _label in label:
                if _label is None:
                    self.label.append(Empty())
                elif isinstance(_label, Label):
                    self.label.append(_label.copy())
                elif is_hashable(_label):
                    self.label.append(ArbitraryHashable(_label))
                else:
                    raise TypeError(f"invalid label type {type(_label)}")

        super().__init__()

    @abstractclassmethod
    def from_numpy(cls, ndarray: np.ndarray, label: Label = None):
        """
            initialized from a numpy ndarray based representation of the objects
        """
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """
            represent the annotation as numpy ndarrays, the ndarrays should be copied,
            and the representation should be symmetric as cls.from_numpy
        """
        pass

    def check(self) -> None:
        """
        Check whether the image/annotation is valid.
        AssertationError is raised if any.

        In subclass, ignore it if no checking is needed.
        """
        pass

    def copy(self):
        """
        Deep copy of all attributes. Return the same type of object.

        In subclass, ignore it if deepcopy(self) is a proper way.
        """
        return deepcopy(self)

    @abstractmethod
    def check_within_image(self) -> IsWithinImage:
        """
        Check whether the annotation elements are all/partly/not within the image.
        Return one of IsWithinImage.YES/NO/PARTIAL.

        In subclass, it must be implemented.
        """
        pass

    def __repr__(self):
        return str(type(self))

    # properties

    @abstractproperty
    def img_w(self) -> int:
        """
        width of the base image

        In subclass, it must be implemented.
        """
        pass

    @abstractproperty
    def img_h(self) -> int:
        """
        height of the base image

        In subclass, it must be implemented.
        """
        pass

    # transformations

    def clip(self) -> IsWithinImage:
        """
        Clip the shapes to make every shape that IsWithinImage.PARTIAL to IsWithinImage.YES.
        If the shape was NO, it remains unchanged.

        In subclass, _clip must be implemented.
        """
        return self._clip()

    @abstractmethod
    def _clip(self):
        pass

    def pad(
            self, up: int = 0, down: int = 0, left: int = 0, right: int = 0,
            fill_value: Union[int, float, np.ndarray] = 0
            ):
        """
            pad the image by the four sides

            In subclass, _pad must be implemented.

        Args:
            up (int, optional): the head of the vertical axis(height). Defaults to 0.
            down (int, optional): the tail of the vertical axis(height). Defaults to 0.
            left (int, optional): the head of the horizontal axis(width). Defaults to 0.
            right (int, optional): the head of the horizontal axis(width). Defaults to 0.
            fill_value (Union[int, float, np.ndarray] , optional):
                the value to fill the padded area. If is a nd.array, the shape must be (self.depth,)
                Defaults to int 0.

        Returns:
            ImageAnnotation: The transformed image/annotation.
        """
        assert (up >= 0 and down >= 0 and left >= 0 and right >= 0),\
            f"Padding length {(up, down, left, right)} should all be non-negative."
        return self._pad(up, down, left, right, fill_value)

    @abstractmethod
    def _pad(self, up: int, down: int, left: int, right: int, fill_value: Union[int, float, np.ndarray]):
        pass

    def crop(self, xmin: int = None, xmax: int = None, ymin: int = None, ymax: int = None):
        """
            Crop the image by the subimage area of (xmin, xmax, ymin, ymax) (inclusive).
            The resulted cropped image is of size width=(xmax-xmin+1) height=(ymax-ymin+1).

            In subclass, _crop must be implemented.

        Args:
            xmin (int, optional): left side of the horizontal axis. Defaults to 0.
            xmax (int, optional): right side of the horizontal axis. Defaults to self.img_w - 1.
            ymax (int, optional): up side of the vertical axis. Defaults to 0.
            ymax (int, optional): down side of the vertical axis. Defaults to self.img_h - 1.

        Returns:
            ImageAnnotation: The transformed image.
        """
        xmin, xmax, ymin, ymax = [
            default_v if v is None else v
            for v, default_v in zip((xmin, xmax, ymin, ymax), (0, self.img_w - 1, 0, self.img_h - 1))
        ]
        assert (0 <= xmin < xmax < self.img_w),\
            f"Invalid horizontal crop length: xmin={xmin} xmax={xmax} img_w={self.img_w}."
        assert (0 <= ymin < ymax < self.img_h),\
            f"Invalid vertical crop length: ymin={ymin} ymax={ymax} img_h={self.img_h}."
        return self._crop(xmin, xmax, ymin, ymax)

    @abstractmethod
    def _crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        pass

    def horizontal_flip(self):
        """
        Flip the image and the shapes horizontally (along the y-axis).

        In subclass, _horizontal_flip must be implemented.
        """
        return self._horizontal_flip()

    @abstractmethod
    def _horizontal_flip(self):
        pass

    def vertical_flip(self):
        """
        Flip the image and the shapes vertically (along the x-axis).

        In subclass, _vertical_flip must be implemented.
        """
        return self._vertical_flip()

    @abstractmethod
    def _vertical_flip(self):
        pass

    def rotate(self, angle: float):
        """
            Rotate the image and the shapes. After rotation, the original image is tightly
            contained in the rotated image, and the padded regions at the four corners are
            filled with 0.

            In subclass, _rotate must be implemented.

        Args:
            angle (float): anti-clockwise rotation angle in radius

        Returns:
            ImageAnnotation: the transformed image
        """
        angle = normalize(angle)
        return self._rotate(angle)

    @abstractmethod
    def _rotate(self, angle: float):
        pass

    def rotate_right_angle(self, angle: int):
        """
        Rotate the image by a multiple of right angle:
        -90, 0, 90, 180, 270, 360...

        In subclass, if _rotate_right_angle is not overidden, it calls:
        if angle == 0:
            return self
        elif angle == 90:
            return self.horizontal_flip().transpose()
        elif angle == 180:
            return self.vertical_flip().horizontal_flip()
        else:
            return self.vertical_flip().transpose()

        Args:
            angle (int): Rotation angle. Must be a multiple of 90.

        Returns:
            ImageAnnotation: the transformed image
        """
        assert angle % 90 == 0, f"Angle must be a multiple of 90, get angle={angle}."
        angle = ((angle // 90) % 4) * 90
        return self._rotate_right_angle(angle)

    def _rotate_right_angle(self, angle: int):
        if angle == 0:
            return self
        elif angle == 90:
            return self.horizontal_flip().transpose()
        elif angle == 180:
            return self.vertical_flip().horizontal_flip()
        else:
            return self.vertical_flip().transpose()

    def rescale(self, factor_x: float, factor_y: float = None):
        """
        Resize the image by scaling factors at x(horizontal)- and y(vertical)- axes.
        In subclass, if _rescale is not overidden, it calls:
            resize(ceil(factor_x * img_w), ceil(factor_y * img_y))

        Args:
            factor_x (float): x-axis scaling factor. Must be greater than 0.
            factor_y (float): y-axis scaling factor. Must be greater than 0.
                If None, use factor_x instead. Default to None.

        Returns:
            ImageAnnotation: the transformed image
        """
        factor_y = factor_x if factor_y is None else factor_y
        assert (factor_x > 0 and factor_y > 0), f"factor_x={factor_x} factor_y={factor_y} must be > 0."
        return self._rescale(factor_x, factor_y)

    def _rescale(self, factor_x: float, factor_y: float):
        target_w = int(ceil(factor_x * self.img_w))
        target_h = int(ceil(factor_y * self.img_h))
        return self.resize(target_w, target_h)

    def resize(self, target_w: int, target_h: int):
        """
            Resize the image to a target width and height.

            In subclass, _resize must be implemented.

        Args:
            dst_w (int): target width. Must be greater than 0.
            dst_h (int): target height. Must be greater than 0.

        Returns:
            ImageAnnotation: the transformed image
        """
        assert (target_w > 0 and target_h > 0),\
            f"Argument target_w={target_w} target_h={target_h} must be > 0."
        return self._resize(target_w, target_h)

    @abstractmethod
    def _resize(self, target_w: int, target_h: int):
        pass

    def transpose(self):
        """
        Transpose the image.

        In subclass, _transpose must be implemented.
        """
        return self._transpose()

    @abstractmethod
    def _transpose(self):
        pass
