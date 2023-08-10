from typing import List, Union, Tuple

import numpy as np

from .basic import ImageAnnotation
from .image import EmptyImage
from .point import Point
from .box import Box
from .polygon import Polygon
from .label import Label
from .utils import IsWithinImage, HALF_PI, PI, ONE_AND_HALF_PI, normalize, vector2angle, rotate_point,\
    rotate_point_right_angle


class OrientedBox(ImageAnnotation):
    def __init__(
            self, x: int, y: int, w: int, h: int,
            theta: float, img_w: int, img_h: int, label: Label = None) -> None:
        """
        Args:
            x (int): x-coordinate of the box center
            y (int): y-coordinate of the box center
            w (int): box width
            h (int): box height
            theta (float): the orientation of the box
            img_w (int): the width of the base image
            img_h (int): the height of the base image
        """
        super().__init__(label)
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.theta = normalize(float(theta))
        self._base_image = EmptyImage(img_w, img_h)
        self.check()

    def check(self):
        assert self.width > 0, f"width must be greater than 0, got {self.width}"
        assert self.height > 0, f"height must be greater than 0, got {self.height}"

    def check_within_image(self):
        if self.xmin >= self.img_w or self.xmax < 0 or self.ymin >= self.img_h or self.ymax < 0:
            return IsWithinImage.NO
        elif self.xmax >= self.img_w or self.xmin < 0 or self.ymax >= self.img_h or self.ymin < 0:
            return IsWithinImage.PARTIAL
        else:
            return IsWithinImage.YES

    @classmethod
    def from_points(cls, points: List[Union[Point, Tuple[float, float]]], img_w, img_h):
        """
            Produce an enclosing oriented box from a quadrilateral
            that consists of four points: top-left, bottom-left,
            bottome-right, top-right. The oritenation is given to
            the up-side.
        """
        assert len(points) == 4
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = [(p.x, p.y) if isinstance(p, Point) else p for p in points]

        dx = (x1 + x3 - x0 - x2)/4
        dy = (y1 + y3 - y0 - y2)/4
        x0 += dx
        x1 -= dx
        x2 += dx
        x3 -= dx
        y0 += dy
        y1 -= dy
        y2 += dy
        y3 -= dy

        theta = vector2angle(x0 - x1, y0 - y1)
        w = np.sqrt((x0 - x3)**2 + (y0 - y3)**2)
        h = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        x = (x0 + x2)/2
        y = (y0 + y2)/2

        return cls(x, y, w, h, theta, img_w, img_h, points[0].label)

    def to_points(self):
        """
            top-left, bottom-left, bottom-right, top-right points
        """
        sin_theta, cos_theta = np.sin(self.theta), np.cos(self.theta)
        half_h, half_w = self.height/2, self.width/2

        center_to_up_dx = cos_theta * half_h
        center_to_up_dy = sin_theta * half_h
        center_to_left_dx = sin_theta * half_w
        center_to_left_dy = - cos_theta * half_w

        return (
            Point(
                self.x + center_to_up_dx + center_to_left_dx,
                self.y + center_to_up_dy + center_to_left_dy,
                self.img_w,
                self.img_h,
                self.label
            ),
            Point(
                self.x - center_to_up_dx + center_to_left_dx,
                self.y - center_to_up_dy + center_to_left_dy,
                self.img_w,
                self.img_h,
                self.label
            ),
            Point(
                self.x - center_to_up_dx - center_to_left_dx,
                self.y - center_to_up_dy - center_to_left_dy,
                self.img_w,
                self.img_h,
                self.label
            ),
            Point(
                self.x + center_to_up_dx - center_to_left_dx,
                self.y + center_to_up_dy - center_to_left_dy,
                self.img_w,
                self.img_h,
                self.label
            )
        )

    def to_box(self) -> Box:
        """
        Get the box that closely encloses this oriented box.
        """
        return Box(
            xmin=self.xmin, xmax=self.xmax, ymin=self.ymin,
            ymax=self.ymax, img_w=self.img_w, img_h=self.img_h,
            label=self.label
        )

    def to_polygon(self) -> Polygon:
        return Polygon(self.to_points(), self.img_w, self.img_h, self.label)

    @classmethod
    def from_numpy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.shape == (5, )
        return cls(array[0], array[1], array[2], array[3], array[4], img_w, img_h, label)

    def to_numpy(self):
        return np.array([self.x, self.y, self.w, self.h, self.theta])

    # properties

    @property
    def width(self):
        return self.w

    @property
    def height(self):
        return self.h

    @property
    def horizontal_width(self):
        return abs(self.h * np.sin(self.theta)) + abs(self.w * np.cos(self.theta))

    @property
    def vertical_height(self):
        return abs(self.w * np.sin(self.theta)) + abs(self.h * np.cos(self.theta))

    @property
    def center_x(self):
        return self.x

    @property
    def center_y(self):
        return self.y

    @property
    def aspect_ratio(self):
        return self.width/self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def xmin(self):
        return self.center_x - self.horizontal_width/2

    @property
    def xmax(self):
        return self.center_x + self.horizontal_width/2

    @property
    def ymin(self):
        return self.center_y - self.vertical_height/2

    @property
    def ymax(self):
        return self.center_y + self.vertical_height/2

    @property
    def img_w(self):
        return self._base_image.img_w

    @property
    def img_h(self):
        return self._base_image.img_h

    def __repr__(self):
        return f"OrientedBox(img_w={self.img_w}, img_h={self.img_h}, w={self.w}, h={self.h}, "\
            f"x={self.x}, y={self.y}, theta={self.theta: .4f}, label={self.label}"

    # transforms

    def _clip(self):
        if self.check_within_image() == IsWithinImage.PARTIAL:
            raise NotImplementedError(f"{type(self)} partially within the image cannot be clipped for now.")
        else:
            return self

    def _pad(self, up, down, left, right, fill_value=None):
        timage = self._base_image.pad(up, down, left, right, fill_value)
        return OrientedBox(
            x=self.x + left,
            y=self.y + up,
            w=self.w,
            h=self.h,
            theta=self.theta,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        timage = self._base_image.crop(xmin, xmax, ymin, ymax)
        return OrientedBox(
            x=self.x - xmin,
            y=self.y - ymin,
            w=self.w,
            h=self.h,
            theta=self.theta,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label
        )

    def _horizontal_flip(self):
        return OrientedBox(
            x=self.img_w - self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            theta=PI - self.theta,
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label
        )

    def _vertical_flip(self):
        return OrientedBox(
            x=self.x,
            y=self.img_h - self.y,
            w=self.w,
            h=self.h,
            theta=-self.theta,
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label
        )

    def _rotate(self, angle):
        rx, ry = rotate_point(self.img_w, self.img_h, angle, self.x, self.y)
        timage = self._base_image.rotate(angle)
        return OrientedBox(
            x=rx,
            y=ry,
            w=self.w,
            h=self.h,
            theta=self.theta + angle,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label
        )

    def _rotate_right_angle(self, right_angle):
        rx, ry = rotate_point_right_angle(self.img_w, self.img_h, right_angle, self.x, self.y)
        if right_angle == 90:
            angle = HALF_PI
        elif right_angle == 180:
            angle = PI
        elif right_angle == 270:
            angle = ONE_AND_HALF_PI
        else:
            angle = 0.

        timage = self._base_image.rotate_right_angle(right_angle)
        return OrientedBox(
            x=rx,
            y=ry,
            w=self.w,
            h=self.h,
            theta=self.theta + angle,
            img_w=timage.img_w,
            img_h=timage.img_h,
            label=self.label
        )

    def _resize(self, dst_w, dst_h):
        '''
            Resize box and img size.
            Note that after resizing/rescaling, a non-horizontal box will be a parallelogram,
            so we only keep the enclosing oriented box of this parallelogram
            (by OrientedBox.from_points)
        '''
        p0, p1, p2, p3 = self.to_points()
        factor_x, factor_y = dst_w/self.img_w, dst_h/self.img_h
        return OrientedBox.from_points(
            points=[
                Point(p0.x * factor_x, p0.y * factor_y, dst_w, dst_h, self.label),
                Point(p1.x * factor_x, p1.y * factor_y, dst_w, dst_h, self.label),
                Point(p2.x * factor_x, p2.y * factor_y, dst_w, dst_h, self.label),
                Point(p3.x * factor_x, p3.y * factor_y, dst_w, dst_h, self.label),
            ],
            img_w=dst_w, img_h=dst_h,
        )

    def _transpose(self):
        return OrientedBox(
            x=self.y,
            y=self.x,
            w=self.w,
            h=self.h,
            theta=HALF_PI - self.theta,
            img_w=self.img_h,
            img_h=self.img_w,
            label=self.label
        )

    # relations

    def __eq__(self, obox):
        if not isinstance(obox, OrientedBox):
            raise TypeError()
        return self.img_w == obox.img_w and\
            self.img_h == obox.img_h and\
            self.x == obox.x and\
            self.y == obox.y and\
            self.w == obox.w and\
            self.h == obox.h and\
            (self.theta - obox.theta) < 1e-5 and\
            self.label == obox.label

    def intersection_area(self, box):
        return self.to_polygon().intersection_area(box.to_polygon())

    def union_area(self, box):
        return self.to_polygon().union_area(box.to_polygon())

    def iou(self, box):
        return self.to_polygon().iou(box.to_polygon())
