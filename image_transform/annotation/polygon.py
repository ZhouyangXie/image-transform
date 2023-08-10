from typing import List, Union, Tuple

import numpy as np
from cv2 import fillPoly, LINE_AA

from .basic import ImageAnnotation
from .image import EmptyImage
from .mask import Mask
from .point import Point, line_intersection
from .utils import IsWithinImage, manhattan_dist
from .label import Empty, Scoped, Label


class Polygon(ImageAnnotation):
    def __init__(
        self,
        points: Union[List[Point], List[Tuple[int, int]]],
        img_w: int,
        img_h: int,
        label: Label = None,
    ) -> None:
        super().__init__(label)
        self._base_image = EmptyImage(img_w, img_h)
        self.points = [
            p if isinstance(p, Point) else Point(p[0], p[1], img_w, img_h)
            for p in points
        ]
        self.check()

    def check(self):
        assert len(self.points) >= 3, "a polygon should have more than two vertices"
        assert Polygon.check_polygon_validity(self.points), "the vertices do not make a valid polygon"
        assert all(p.img_w == self.img_w and p.img_h == self.img_h for p in self.points),\
            "the points are not associated with the image of the same size"

    def check_within_image(self):
        is_within_image = [p.check_within_image() == IsWithinImage.YES for p in self.points]
        if all(is_within_image):
            return IsWithinImage.YES
        elif any(is_within_image):
            return IsWithinImage.PARTIAL
        else:
            return IsWithinImage.NO

    # conversions

    def to_mask(self, dtype=np.int32) -> Mask:
        """
        Convert the Polygon object to a Mask object.
        self.label must be Empty or Scoped.

        Args:
            dtype (optional): one of an integer data type or bool. Defaults to bool.
        """
        assert isinstance(self.label, (Empty, Scoped)), "Only Empty or Scoped label can be converted to Mask"
        value = 1 if isinstance(self.label, Empty) else 1 + self.label.scope.index(self.label.value)
        mask = np.zeros((self.img_h, self.img_w), dtype=dtype)
        mask = fillPoly(mask, pts=[np.array([(p.x, p.y) for p in self.points])], color=value, lineType=LINE_AA)
        return Mask(mask, self.label)

    @staticmethod
    def check_polygon_validity(points: List[Point]) -> bool:
        N = len(points)
        for i in range(N):
            for j in range(N):
                if abs(i - j) in (0, 1, N - 1):
                    continue
                if (line_intersection((points[i], points[(i + 1) % N]), (points[j], points[(j + 1) % N])) is not None):
                    return False

        return True

    @classmethod
    def from_numpy(cls, array: np.ndarray, img_w: int, img_h: int, label: Label = None):
        assert array.ndim == 2 and array.shape[1] == 2
        return cls(array, img_w, img_h, label)

    def to_numpy(self):
        return np.stack([p.to_numpy() for p in self.points])

    # properties

    @property
    def xmin(self) -> int:
        return min([p.x for p in self.points])

    @property
    def xmax(self) -> int:
        return max([p.x for p in self.points])

    @property
    def ymin(self) -> int:
        return min([p.y for p in self.points])

    @property
    def ymax(self):
        return max([p.y for p in self.points])

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return abs(self.signed_area)

    @property
    def img_w(self):
        return self._base_image.width

    @property
    def img_h(self):
        return self._base_image.height

    @staticmethod
    def _get_signed_area(points: List[Point]) -> float:
        assert len(points) > 2
        N = len(points)
        s = 0
        for i in range(N):
            s += (points[i].x * points[(i + 1) % N].y - points[i].y * points[(i + 1) % N].x)

        return s / 2

    @property
    def signed_area(self):
        return Polygon._get_signed_area(self.points)

    def __repr__(self):
        s = f"Polygon(img_w={self.img_w}, img_h={self.img_h}, label={self.label}, x=..., y=...):\n"
        for p in self.points:
            s += f"({p.x}, {p.y}),\n"
        return s

    # transformations

    def _clip(self):
        if self.check_within_image() == IsWithinImage.PARTIAL:
            # TODO: a feasible implementation is to just move outside points to image boundary
            raise NotImplementedError(f"{type(self)} partially within the image cannot be clipped for now.")
        else:
            return self

    def _pad(self, up, down, left, right, fill_value=None):
        timage = self._base_image.pad(up, down, left, right, fill_value)
        img_w, img_h = timage.img_w, timage.img_h
        return Polygon(
            points=[p.pad(up, down, left, right, fill_value) for p in self.points],
            img_w=img_w,
            img_h=img_h,
            label=self.label,
        )

    def _crop(self, xmin, xmax, ymin, ymax):
        timage = self._base_image.crop(xmin, xmax, ymin, ymax)
        img_w, img_h = timage.img_w, timage.img_h
        return Polygon(
            points=[p.crop(xmin, xmax, ymin, ymax) for p in self.points],
            img_w=img_w,
            img_h=img_h,
            label=self.label,
        )

    def _horizontal_flip(self):
        return Polygon(
            points=[p.horizontal_flip() for p in self.points],
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    def _vertical_flip(self):
        return Polygon(
            points=[p.vertical_flip() for p in self.points],
            img_w=self.img_w,
            img_h=self.img_h,
            label=self.label,
        )

    def _rotate(self, angle):
        timage = self._base_image.rotate(angle)
        img_w, img_h = timage.img_w, timage.img_h
        return Polygon(
            points=[p.rotate(angle) for p in self.points],
            img_w=img_w,
            img_h=img_h,
            label=self.label,
        )

    def _rotate_right_angle(self, right_angle):
        timage = self._base_image.rotate_right_angle(right_angle)
        img_w, img_h = timage.img_w, timage.img_h
        return Polygon(
            points=[p.rotate_right_angle(right_angle) for p in self.points],
            img_w=img_w,
            img_h=img_h,
            label=self.label,
        )

    def _resize(self, dst_w, dst_h):
        return Polygon(
            points=[p.resize(dst_w, dst_h) for p in self.points],
            img_w=dst_w,
            img_h=dst_h,
            label=self.label,
        )

    def _transpose(self):
        return Polygon(
            points=[p.transpose() for p in self.points],
            img_w=self.img_h,
            img_h=self.img_w,
            label=self.label,
        )

    # relation

    def intersection_area(self, other):
        assert isinstance(other, Polygon),\
            "currently I/U/IoU computation is only supported for Box-Box or BoxArray-BoxArray"
        assert self.img_w == other.img_w and self.img_h == other.img_h,\
            f"base image shape must be equal: self({self.img_w}, {self.img_h}) v.s. box({other.img_w}, {other.img_h})"

        # TODO: improve readability and conciseness of the code of this complicated algorithm
        P = self
        Q = other
        P_signed_area = P.signed_area
        Q_signed_area = Q.signed_area
        if P_signed_area * Q_signed_area > 0:
            Q = Polygon(Q.points[::-1], img_w=Q.img_w, img_h=Q.img_h)
            Q_signed_area = -Q_signed_area

        # find all intersection points
        # and the edges that intersec
        N, M = len(P.points), len(Q.points)
        # the index of the points after which the intersection point lie
        P_inter_index = []
        Q_inter_index = []
        inter_points = []
        for i in range(N):
            for j in range(M):
                inter_point = line_intersection(
                    (P.points[i], P.points[(i + 1) % N]),
                    (Q.points[j], Q.points[(j + 1) % M]),
                )
                if inter_point is not None:
                    P_inter_index.append(i)
                    Q_inter_index.append(j)
                    inter_points.append(inter_point)

        # handle zero intersection case
        L = len(inter_points)
        if L == 0:
            num_inter_q0_P = sum(
                [
                    0
                    if line_intersection(
                        (Point(P.xmin - 11, P.ymin - 8, P.img_w, P.img_h), Q.points[0]),
                        (P.points[i], P.points[(i + 1) % N]),
                    )
                    is None
                    else 1
                    for i in range(len(P.points))
                ]
            )
            num_inter_p0_Q = sum(
                [
                    0
                    if line_intersection(
                        (Point(Q.xmin - 11, Q.ymin - 8, P.img_w, P.img_h), P.points[0]),
                        (Q.points[i], Q.points[(i + 1) % M]),
                    )
                    is None
                    else 1
                    for i in range(len(Q.points))
                ]
            )
            if num_inter_q0_P % 2 == 1 or num_inter_p0_Q % 2 == 1:
                return min(abs(Q_signed_area), abs(P_signed_area))
            else:
                return 0.0

        # assert L % 2 == 0

        # insert the index of the intersection points into both polygons
        P_new_points = []
        for i, p in enumerate(P.points):
            P_new_points.append(p)
            if i in P_inter_index:
                i_inter_point_inds = [
                    j for j, ind in enumerate(P_inter_index) if ind == i
                ]
                i_inter_point_inds = sorted(
                    i_inter_point_inds,
                    key=lambda k: manhattan_dist(inter_points[k], p),
                )
                P_new_points.extend(i_inter_point_inds)

        Q_new_points = []
        for i, q in enumerate(Q.points):
            Q_new_points.append(q)
            if i in Q_inter_index:
                i_inter_point_inds = [
                    j for j, ind in enumerate(Q_inter_index) if ind == i
                ]
                i_inter_point_inds = sorted(
                    i_inter_point_inds,
                    key=lambda k: manhattan_dist(inter_points[k], q),
                )
                Q_new_points.extend(i_inter_point_inds)

        s = 0
        visited = [False] * L
        while not all(visited):
            # find a start point
            start_inter_ind = min([i for i, v in enumerate(visited) if not v])
            visited[start_inter_ind] = True

            # start with P
            points = [inter_points[start_inter_ind]]
            cur_index = P_new_points.index(start_inter_ind)
            cur_list = P_new_points
            while True:
                cur_index = (cur_index + 1) % len(cur_list)
                cur_point = cur_list[cur_index]

                if isinstance(cur_point, Point):
                    points.append(cur_point)
                elif isinstance(cur_point, int):
                    if cur_point == start_inter_ind:
                        break
                    else:
                        points.append(cur_point)
                        visited[cur_point] = True
                        if id(cur_list) == id(Q_new_points):
                            cur_list = P_new_points
                        else:
                            cur_list = Q_new_points
                        cur_index = cur_list.index(cur_point)
                else:
                    raise RuntimeError

            points = [inter_points[p] if isinstance(p, int) else p for p in points]
            s += abs(Polygon._get_signed_area(points))

            # start with Q
            points = [inter_points[start_inter_ind]]
            cur_index = Q_new_points.index(start_inter_ind)
            cur_list = Q_new_points
            while True:
                cur_index = (cur_index + 1) % len(cur_list)
                cur_point = cur_list[cur_index]

                if isinstance(cur_point, Point):
                    points.append(cur_point)
                elif isinstance(cur_point, int):
                    if cur_point == start_inter_ind:
                        break
                    else:
                        points.append(cur_point)
                        visited[cur_point] = True
                        if id(cur_list) == id(Q_new_points):
                            cur_list = P_new_points
                        else:
                            cur_list = Q_new_points
                        cur_index = cur_list.index(cur_point)
                else:
                    raise RuntimeError

            points = [inter_points[p] if isinstance(p, int) else p for p in points]
            s += abs(Polygon._get_signed_area(points))

        return (abs(P_signed_area) + abs(Q_signed_area) - s) / 2

    def union_area(self, other):
        return self.area + other.area - self.intersection_area(other)

    def iou(self, other):
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        else:
            return inter / (self.area + other.area - inter)

    def __eq__(self, polygon):
        if not isinstance(polygon, Polygon):
            raise TypeError()

        if len(polygon.points) != len(self.points) or\
                polygon.img_w != self.img_w or\
                polygon.img_h != self.img_h or\
                polygon.label != self.label:
            return False

        L = len(polygon.points)

        start = None
        for i, point in enumerate(self.points):
            if point == polygon.points[0]:
                start = i
                break

        if start is None:
            return False

        for i, j in enumerate(range(start, L)):
            if polygon.points[i] != self.points[j]:
                return False

        for i, j in enumerate(range(L - start, L)):
            if self.points[i] != polygon.points[j]:
                return False

        return True
