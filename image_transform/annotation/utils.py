from typing import Union, Tuple
from enum import Enum, auto

import numpy as np
import cv2

HALF_PI = np.pi/2
PI = np.pi
ONE_AND_HALF_PI = 1.5 * np.pi
TWO_PI = 2 * np.pi


class IsWithinImage(Enum):
    YES = auto()
    PARTIAL = auto()
    NO = auto()


def vector2angle(dx: Union[float, int], dy: Union[float, int]) -> float:
    """
        compute the orientation of a vector (dx, dy) in radian
    """
    dx = max(dx, 1e-4) if dx >= 0 else min(dx, -1e-4)
    theta = np.arctan(dy/dx)
    if dx > 0:
        if dy < 0:
            theta += TWO_PI
    else:
        theta += PI
    return theta


def normalize(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Normalize a radian angle to [2, 2*pi)

    Args:
        theta (Union[float, np.ndarray]): angle(s)

    Returns:
        Union[float, np.ndarray]: normalized angle(s)
    """
    p = theta/TWO_PI
    p -= np.floor(p)
    return p * TWO_PI


def manhattan_dist(p1, p2):
    return np.abs(p1.x - p2.x) + np.abs(p1.y - p2.y)


def euclidean_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_rotated_image_size(w: int, h: int, a: float) -> Tuple[int, int]:
    rw = abs(w*np.cos(a))+abs(h*np.sin(a))
    rh = abs(h*np.cos(a))+abs(w*np.sin(a))
    return int(np.floor(rw)), int(np.floor(rh))


def rotate_point(w: int, h: int, a: float, x: Union[int, np.ndarray], y: Union[int, np.ndarray])\
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[int, int]]:
    if isinstance(x, int):
        assert isinstance(y, int)
    else:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    a = normalize(a)
    sin_a, cos_a = np.sin(a), np.cos(a)
    rx_to_o, ry_to_o = x * cos_a - y * sin_a, x * sin_a + y * cos_a
    rw, rh = get_rotated_image_size(w, h, a)

    if a <= HALF_PI:
        o_offset_x = h * sin_a
        o_offset_y = 0
    elif a <= PI:
        o_offset_x = rw
        o_offset_y = - h * cos_a
    elif a <= ONE_AND_HALF_PI:
        o_offset_x = - w * cos_a
        o_offset_y = rh
    else:
        o_offset_x = 0
        o_offset_y = - w * sin_a

    if isinstance(x, int):
        return int(rx_to_o + o_offset_x), int(ry_to_o + o_offset_y)
    else:
        return np.array(rx_to_o + o_offset_x, dtype=int), np.array(ry_to_o + o_offset_y, dtype=int)


def rotate_point_right_angle(
        img_w: int, img_h: int, right_angle: float,
        x: Union[int, np.ndarray], y: Union[int, np.ndarray])\
            -> Union[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    if right_angle == 90:
        return img_h - y, x
    elif right_angle == 180:
        return img_w - x, img_h - y
    elif right_angle == 270:
        return y, img_w - x
    else:
        return x, y


def nms(duplicate_matrix: np.ndarray) -> np.ndarray:
    """
    Non-maximum Suppression on arbitrary targets.
    Assume target 0 ~ (N-1) are sorted by confidence in descending order.

    Args:
        duplicate_matrix (np.ndarray): Shape (N, N). N is the number of target.
            Non-zero value at (i, j) means target j is a duplicate of target i.
            Must be a symmetric matrix and diagnal entries are non-zero.

    Returns:
        np.ndarray: (N, ) boolean array. True is kept, False is duplicate to be removed.
    """
    assert duplicate_matrix.ndim == 2, f"Got shape {duplicate_matrix.shape}"
    N, _N = duplicate_matrix.shape
    assert N == _N, "Must be a square matrix. Got shape {duplicate_matrix.shape}."
    M = duplicate_matrix != 0
    assert np.all(np.diag(M)), "Diagonal entries must be non-zero."

    preserved = np.zeros(N, dtype=bool)
    for i in range(N):
        if ~M[i, i]:
            continue
        else:
            preserved[i] = True
        duplicate = M[i, :]
        M[duplicate, :] = 0
        M[:, duplicate] = 0

    return preserved


def sequence2rle(sequence):
    if not np.issubdtype(sequence.dtype, np.integer):
        sequence = sequence.astype(bool)
    sequence = sequence.flatten()
    positions_of_change = np.diff(sequence).nonzero()[0] + 1
    positions_of_change = np.concatenate(([0], positions_of_change, [len(sequence)]))
    segment_lengths = np.diff(positions_of_change)
    segment_values = sequence[positions_of_change[:-1]]
    return segment_lengths, segment_values, sequence.dtype


def rle2sequence(rle):
    segment_lengths, segment_values, dtype = rle
    sequence = np.zeros(segment_lengths.sum(), segment_values.dtype)
    cur = 0
    for length, value in zip(segment_lengths, segment_values):
        sequence[cur:(cur + length)] = value
        cur += length

    return sequence.astype(dtype)


def pad_image(image: np.ndarray, up: int, down: int, left: int, right: int, fill_value):
    if image.ndim == 2:
        squeezed = True
        image = image.reshape((*image.shape, 1))
    else:
        squeezed = False
        assert image.ndim == 3

    h, w, c = image.shape
    h_padded, w_padded = h + up + down, w + left + right

    padded_data = np.empty(shape=(h_padded, w_padded, c), dtype=image.dtype)
    if np.isscalar(fill_value):
        padded_data.fill(fill_value)
    else:
        assert fill_value.shape == (c,),\
            f"image data channel is {c}, incompatible with the fill-value {fill_value}"
        padded_data[..., :] = fill_value
    padded_data[up:(up + h), left:(left + w), :] = image

    if squeezed:
        assert c == 1
        padded_data = padded_data.reshape((h_padded, w_padded))

    return padded_data


def rotate_image(image: np.ndarray, a: float, interpolation_flag: int = cv2.INTER_NEAREST):
    assert image.dtype in (np.uint8, np.uint16, np.float32, np.float64)
    w, h = image.shape[1], image.shape[0]
    rw, rh = get_rotated_image_size(w, h, a)
    points = [(w, 0), (w, h), (0, h)]
    rpoints = [rotate_point(w, h, a, x, y) for x, y in points]
    M = cv2.getAffineTransform(
        src=np.array(points, dtype=np.float32),
        dst=np.array(rpoints, dtype=np.float32),
    )
    return cv2.warpAffine(image, M, (rw, rh), interpolation_flag)
