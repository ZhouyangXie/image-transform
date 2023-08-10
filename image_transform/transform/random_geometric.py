from random import random, choice, randint
from math import pi, ceil

from .basic import EmptyTransform
from .geometric import HorizontalFlip, VerticalFlip, Rotate,\
    RotateRightAngle, Transpose, GeometricTransform, Crop


class RandomHorizontalFlip(GeometricTransform):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        assert prob <= 1 and prob >= 0
        self.prob = prob
        self.is_flipped = None

    def _transform(self, image_annotation):
        self.is_flipped = random() < self.prob
        if self.is_flipped:
            return image_annotation.horizontal_flip()
        else:
            return image_annotation

    def _get_inverse(self):
        if self.is_flipped:
            return HorizontalFlip()
        else:
            return EmptyTransform()


class RandomVerticalFlip(GeometricTransform):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        assert prob <= 1 and prob >= 0
        self.prob = prob
        self.is_flipped = None

    def _transform(self, image_annotation):
        self.is_flipped = random() < self.prob
        if self.is_flipped:
            return image_annotation.vertical_flip()
        else:
            return image_annotation

    def _get_inverse(self):
        if self.is_flipped:
            return VerticalFlip()
        else:
            return EmptyTransform()


class RandomRotate(GeometricTransform):
    def __init__(self):
        super().__init__()
        self.angle = None

    def _transform(self, image_annotation):
        self.angle = random() * 2 * pi
        return image_annotation.rotate(self.angle)

    def _get_inverse(self):
        return Rotate(-self.angle)


class RandomRotateRightAngle(GeometricTransform):
    def __init__(self):
        super().__init__()
        self.angle = None

    def _transform(self, image_annotation):
        self.angle = choice((0, 90, 180, 270))
        return image_annotation.rotate_right_angle(self.angle)

    def _get_inverse(self):
        return RotateRightAngle(-self.angle)


class RandomTranspose(GeometricTransform):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        assert prob <= 1 and prob >= 0
        self.prob = prob
        self.is_transposed = None

    def _transform(self, image_annotation):
        self.is_transposed = random() < self.prob
        if self.is_transposed:
            return image_annotation.transpose()
        else:
            return image_annotation

    def _get_inverse(self):
        if self.is_transposed:
            return Transpose()
        else:
            return EmptyTransform()


class RandomCrop(GeometricTransform):
    def __init__(self, ratio_w=0.5, ratio_h=None, target_w=None, target_h=None):
        if target_w is not None:
            assert isinstance(target_w, int) and target_w > 0, f"target_w must be a positive int, got {target_w}"
            self.use_ratio = False
            self.target_w = target_w
            target_h = target_w if target_h is None else target_h
            assert isinstance(target_h, int) and target_h > 0, f"target_h must be a positive int, got {target_h}"
            self.target_h = target_h
        else:
            assert isinstance(ratio_w, float) and 1 >= ratio_w > 0, f"ratio_w must be a float in [0, 1], got {ratio_w}"
            ratio_h = ratio_w if ratio_h is None else ratio_h
            assert isinstance(ratio_h, float) and 1 >= ratio_h > 0, f"ratio_h must be a float in [0, 1], got {ratio_h}"
            self.use_ratio = True
            self.ratio_w, self.ratio_h = ratio_w, ratio_h

    def _transform(self, image_annotation):
        src_w, src_h = image_annotation.img_w, image_annotation.img_h
        if self.use_ratio:
            target_w, target_h = int(ceil(self.ratio_w * src_w)), int(ceil(self.ratio_h * src_h))
        else:
            target_w, target_h = self.target_w, self.target_h
        assert target_w <= src_w and target_h <= src_h

        w_start = randint(0, src_w - target_w)
        h_start = randint(0, src_h - target_h)

        self._transform = Crop(w_start, w_start + target_w - 1, h_start, h_start + target_h - 1)
        image_annotation = self._transform.transform(image_annotation)
        return image_annotation

    def _get_inverse(self):
        return self._transform.get_inverse()
