from image_transform.annotation import ImageAnnotation
from .basic import Transform, SequentialTransforms
from ..annotation.utils import get_rotated_image_size


class GeometricTransform(Transform):
    """
        Base class for geometric transformations. A subclass needs
        to have _transform implemented.
    """
    pass


class Pad(GeometricTransform):
    """
    Pad the image by the four sides.

    Args:
        up (int): up-side padding size.
        down (int, optional): down-side padding size.
            If None, same as up. Defaults to None.
        left (int, optional): left-side padding size.
            If None, same as down. Defaults to None.
        right (int, optional): right-side padding size.
            If None, same as left. Defaults to None.
        fill_value (int, optional): filling value. Defaults to 0.
    """
    def __init__(
            self, up: int, down: int = None,
            left: int = None, right: int = None, fill_value=0
            ):
        super().__init__()
        assert up >= 0 and down >= 0 and left >= 0 and right >= 0
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.fill_value = fill_value

    def _transform(self, image_annotation):
        return image_annotation.pad(self.up, self.down, self.left, self.right, self.fill_value)

    def _get_inverse(self):
        return Crop(
            xmin=self.left,
            xmax=self.left + self.img_w - 1,
            ymin=self.up,
            ymax=self.up + self.img_h - 1,
        )


class PadTo(GeometricTransform):
    """
    Pad(symmetric) the image to a target size.

    Args:
        target_w (int): target width.
        target_h (int): target height. If None, same as target_w. Defaults to None.
        fill_value (int, optional): filling value. Defaults to 0.
    """
    def __init__(self, target_w: int, target_h: int = None, fill_value=0):
        super().__init__()
        assert target_w > 0 and target_h > 0
        self.target_w = target_w
        self.target_h = target_h
        self.up, self.down, self.left, self.right =\
            None, None, None, None
        self.fill_value = fill_value

    def _transform(self, image_annotation):
        assert self.target_w >= self.img_w and self.target_h >= self.img_h,\
            f"Image width/height({self.img_w, self.img_h}) should both be "
        f"smaller than the target({self.target_w, self.target_h})"
        self.up = (self.target_h - self.img_h)//2
        self.down = self.target_h - self.img_h - self.up
        self.left = (self.target_w - self.img_w)//2
        self.right = self.target_w - self.img_w - self.left
        return image_annotation.pad(self.up, self.down, self.left, self.right, self.fill_value)

    def _get_inverse(self):
        return Crop(
            xmin=self.left,
            xmax=self.left + self.img_w - 1,
            ymin=self.up,
            ymax=self.up + self.img_h - 1,
        )


class PadToMultiple(GeometricTransform):
    """
    Pad(symmetric) the image to the least size so that the width is a
    multiple of w_base, so as height.

    Args:
        w_base (int): as description.
        h_base (int, optional): as description. If None, same as w_base. Defaults to None.
        fill_value (int, optional): filling value. Defaults to 0.
    """
    def __init__(self, w_base: int, h_base: int = None, fill_value=0):
        super().__init__()
        h_base = w_base if h_base is None else h_base
        assert w_base > 0 and h_base > 0
        self.w_base, self.h_base = int(w_base), int(h_base)
        self.fill_value = fill_value
        self.up, self.down, self.left, self.right =\
            None, None, None, None

    def _transform(self, image_annotation):
        residual_w, residual_h = (-self.img_w) % self.w_base, (-self.img_h) % self.h_base
        self.left = residual_w//2
        self.right = residual_w - self.left
        self.up = residual_h//2
        self.down = residual_h - self.up
        return image_annotation.pad(self.up, self.down, self.left, self.right, self.fill_value)

    def _get_inverse(self):
        return Crop(
            xmin=self.left,
            xmax=self.left + self.img_w - 1,
            ymin=self.up,
            ymax=self.up + self.img_h - 1,
        )


class Crop(GeometricTransform):
    """
    Crop the image by the specified window.
    The inverse transformation pad the crop images with filling value 0.

    Args:
        xmin (int): crop window left boundary.
        xmax (int): crop window right boundary. The result crop width is xmax - xmin + 1.
        ymin (int): crop window up boundary.
        ymax (int): crop window down boundary. The result crop height is ymax - ymin + 1.
    """
    def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int):
        super().__init__()
        assert xmin <= xmax and ymin <= ymax
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def _transform(self, image_annotation):
        return image_annotation.crop(self.xmin, self.xmax, self.ymin, self.ymax)

    def _get_inverse(self):
        return Pad(
            left=self.xmin,
            right=self.img_w - self.xmax - 1,
            up=self.ymin,
            down=self.img_h - self.ymax - 1,
            fill_value=0
        )


class CentralCrop(GeometricTransform):
    """
    Crop the image at the center.
    The inverse transformation pad the crop images with filling value 0.

    Args:
        target_w (int): crop window width.
        target_h (int): crop window height.
    """
    def __init__(self, target_w: int, target_h: int):
        super().__init__()
        assert target_w > 0 and target_h > 0
        self.target_w = target_w
        self.target_h = target_h
        self.xmin, self.xmax, self.ymin, self.ymax =\
            None, None, None, None

    def _transform(self, image_annotation):
        assert self.target_w <= self.img_w and self.target_h <= self.img_h,\
            f"Image width/height({self.img_w, self.img_h}) should both be "
        f"greater than the target({self.target_w, self.target_h})"
        self.xmin = (self.img_w - self.target_w)//2
        self.xmax = self.xmin + self.target_w - 1
        self.ymin = (self.img_h - self.target_h)//2
        self.ymax = self.ymin + self.target_h - 1
        return image_annotation.crop(self.xmin, self.xmax, self.ymin, self.ymax)

    def _get_inverse(self):
        return Pad(
            up=self.ymin,
            down=self.img_h - self.ymax - 1,
            left=self.xmin,
            right=self.img_w - self.xmax - 1,
            fill_value=0
        )


class HorizontalFlip(GeometricTransform):
    """
    Horizontal flipping.
    """
    def __init__(self):
        super().__init__()

    def _transform(self, image_annotation):
        return image_annotation.horizontal_flip()

    def _get_inverse(self):
        return HorizontalFlip()


class VerticalFlip(GeometricTransform):
    """
    Vertical flipping.
    """
    def __init__(self):
        super().__init__()

    def _transform(self, image_annotation):
        return image_annotation.vertical_flip()

    def _get_inverse(self):
        return VerticalFlip()


class Rotate(GeometricTransform):
    """
    Rotate the image by an arbitrary angle.

    Args:
        angle (float): anti-clockwise rotation angle in radian.
    """
    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def _transform(self, image_annotation):
        return image_annotation.rotate(self.angle)

    def _get_inverse(self):
        ori_w, ori_h = self.img_w, self.img_h
        r_w, r_h = get_rotated_image_size(ori_w, ori_h, self.angle)
        ir_w, ir_h = get_rotated_image_size(r_w, r_h, -self.angle)
        horizontal_pad = (ir_w - ori_w)//2
        vertical_pad = (ir_h - ori_h)//2

        return SequentialTransforms([
            Rotate(-self.angle),
            Crop(
                xmin=horizontal_pad,
                xmax=horizontal_pad + ori_w - 1,
                ymin=vertical_pad,
                ymax=vertical_pad + ori_h - 1
            )
        ])


class RotateRightAngle(GeometricTransform):
    """
    Rotate the image anti-clockwise by a multiple of right angle:
    -90, 0, 90, 180, 270, 360...

    Args:
        angle (int): Rotation angle. Must be a multiple of 90.
    """
    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def _transform(self, image_annotation):
        return image_annotation.rotate_right_angle(self.angle)

    def _get_inverse(self):
        return RotateRightAngle(-self.angle)


class Rescale(GeometricTransform):
    """
    Resize the image by scaling factors at x(horizontal)- and y(vertical)- axes.

    Args:
        factor_x (float): x-axis scaling factor. Must be greater than 0.
        factor_y (float): y-axis scaling factor. Must be greater than 0.
            If None, use factor_x instead. Default to None.
    """
    def __init__(self, factor_x, factor_y=None):
        super().__init__()
        self.factor_x = factor_x
        self.factor_y = factor_y

    def _transform(self, image_annotation):
        return image_annotation.rescale(self.factor_x, self.factor_y)

    def _get_inverse(self):
        # use Rescale to perform the inverse is not as accurate as Resize
        return Resize(self.img_w, self.img_h)


class Resize(GeometricTransform):
    """
    Resize the image to a target width and height.

    Args:
        dst_w (int): target width. Must be greater than 0.
        dst_h (int): target height. Must be greater than 0.
    """
    def __init__(self, dst_w, dst_h, keep_aspect_ratio: bool = False, fill_value: int = 0):
        super().__init__()
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.keep_aspect_ratio = keep_aspect_ratio
        self.fill_value = fill_value
        self.pad_transform = None

    def _transform(self, image_annotation):
        if self.keep_aspect_ratio:
            src_aspect_ratio = self.img_w/self.img_h
            dst_aspect_ratio = self.dst_w/self.dst_h

            if src_aspect_ratio > dst_aspect_ratio:
                dst_h = int(self.dst_w/src_aspect_ratio)
                image_annotation = image_annotation.resize(self.dst_w, dst_h)
                self.pad_transform = Pad(up=0, down=max(0, self.dst_h - dst_h), left=0, right=0, fill_value=self.fill_value)
            else:
                dst_w = int(self.dst_h * src_aspect_ratio)
                image_annotation = image_annotation.resize(dst_w, self.dst_h)
                self.pad_transform = Pad(up=0, down=0, left=0, right=max(0, self.dst_w - dst_w), fill_value=self.fill_value)

            image_annotation = self.pad_transform.transform(image_annotation)
        else:
            image_annotation = image_annotation.resize(self.dst_w, self.dst_h)

        return image_annotation

    def _get_inverse(self):
        if self.keep_aspect_ratio:
            return SequentialTransforms([
                self.pad_transform.get_inverse(),
                Resize(self.img_w, self.img_h)
            ])
        else:
            return Resize(self.img_w, self.img_h)


class Transpose(GeometricTransform):
    """
        Transpose the image.
    """
    def __init__(self):
        super().__init__()

    def _transform(self, image_annotation):
        return image_annotation.transpose()

    def _get_inverse(self):
        return Transpose()


class ResizeAndPad(GeometricTransform):
    """
        Resize the longer side of the image to `dst_size` and pad the shorter side to a square
    """
    def __init__(self, dst_size: int):
        super().__init__()
        assert dst_size > 0
        self.dst_size = dst_size
        self.resize_and_pad = None

    def _transform(self, image: ImageAnnotation):
        ori_w, ori_h = image.img_w, image.img_h
        if ori_w >= ori_h:
            dst_w = self.dst_size
            dst_h = int(ori_h * dst_w/ori_w)
            pad_left, pad_right = 0, 0
            pad_up = (dst_w - dst_h)//2
            pad_down = dst_w - dst_h - pad_up
        else:
            dst_h = self.dst_size
            dst_w = int(ori_w * dst_h/ori_h)
            pad_up, pad_down = 0, 0
            pad_left = (dst_h - dst_w)//2
            pad_right = dst_h - dst_w - pad_left

        self.resize_and_pad = SequentialTransforms([
            Resize(dst_w, dst_h), Pad(pad_up, pad_down, pad_left, pad_right)
        ])
        return self.resize_and_pad(image)

    def _get_inverse(self):
        return self.resize_and_pad.get_inverse()
