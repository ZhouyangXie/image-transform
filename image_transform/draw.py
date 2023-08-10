from math import pi
from typing import Union
from random import choices

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.lines as lines

import numpy as np

from .annotation import Composite, Image, EmptyImage, Point, PointArray,\
    Box, BoxArray, OrientedBox, OrientedBoxArray, Polygon, Mask, ImageAnnotation,\
    Label, Empty, Scoped, ScopedWithConfidence, MultipleScoped,\
    ProbabilisticMultipleScoped, ScopedMaskWithConfidence, ArbitraryHashable


_GLOBAL_LINE_WIDTH = 0.5
_POINT_RADIUS = 5
_COLOR_SPACE = list(mpl_colors.XKCD_COLORS.values())


def _get_values_of_interest(label: Label):
    if isinstance(label, Empty):
        return [None]
    elif isinstance(label, (Scoped, ScopedWithConfidence, ArbitraryHashable)):
        return [label.value]
    elif isinstance(label, MultipleScoped):
        return label.values
    elif isinstance(label, ProbabilisticMultipleScoped):
        return [label.to_scoped().value]
    elif isinstance(label, ScopedMaskWithConfidence):
        return label.scope
    else:
        return [None]


def draw_box(ax, box: Box, label2color):
    labels = _get_values_of_interest(box.label)
    for offset, label in enumerate(labels):
        ax.add_patch(
            patches.Rectangle(
                xy=(box.xmin - offset * _GLOBAL_LINE_WIDTH, box.ymin - offset * _GLOBAL_LINE_WIDTH),
                width=box.width + 2 * offset * _GLOBAL_LINE_WIDTH,
                height=box.height + 2 * offset * _GLOBAL_LINE_WIDTH,
                edgecolor=label2color[label],
                fill=False,
                linewidth=_GLOBAL_LINE_WIDTH
            )
        )


def draw_oriented_box(ax, box: OrientedBox, label2color):
    labels = _get_values_of_interest(box.label)
    tl, _, _, tr = box.to_points()
    for offset, label in enumerate(labels):
        ax.add_patch(
            patches.Rectangle(
                xy=(tl.x, tl.y),
                width=box.width + 2 * offset * _GLOBAL_LINE_WIDTH,
                height=box.height + 2 * offset * _GLOBAL_LINE_WIDTH,
                angle=(180 * box.theta)/pi + 90,
                edgecolor=label2color[label],
                fill=False,
                linewidth=_GLOBAL_LINE_WIDTH
            )
        )
    ax.add_patch(
        patches.Arrow(
            x=box.x,
            y=box.y,
            dx=(tl.x + tr.x)/2 - box.x,
            dy=(tl.y + tr.y)/2 - box.y,
            color=label2color[label],
            linewidth=_GLOBAL_LINE_WIDTH
        )
    )


def draw_point(ax, point: Point, label2color):
    labels = _get_values_of_interest(point.label)
    for offset, label in enumerate(labels):
        ax.add_patch(
            patches.Circle(
                xy=(point.x, point.y),
                radius=_POINT_RADIUS + len(labels) - offset - 1,
                fill=True,
                edgecolor=label2color[label],
                facecolor=label2color[label]
            )
        )


def draw_polygon(ax, polygon: Polygon, label2color):
    labels = _get_values_of_interest(polygon.label)
    for offset, label in enumerate(labels):
        ax.add_patch(
            patches.Polygon(
                xy=np.array([
                    (p.x + offset * _GLOBAL_LINE_WIDTH, p.y + offset * _GLOBAL_LINE_WIDTH)
                    for p in polygon.points
                ]),
                edgecolor=label2color[label],
                fill=False,
                linewidth=_GLOBAL_LINE_WIDTH
            )
        )


def draw_mask(ax, mask: Mask, label2color):
    labels = mask.scope
    colors = [mpl_colors.to_rgb(label2color[label]) for label in labels]
    canvas = np.zeros((mask.img_h, mask.img_w, 3), dtype=float)
    for label_i, color in zip(range(1, len(mask.scope) + 1), colors):
        canvas[mask.mask == label_i] = color

    ax.imshow(canvas)


def draw_image(ax, image: Union[Image, EmptyImage]):
    if isinstance(image, EmptyImage):
        image = image.to_image()
    ax.imshow(image.data, cmap='gray')


def generate_colors(num_colors):
    assert num_colors <= len(_COLOR_SPACE),\
        f"The number of colors({num_colors}) surpasses the"\
        f"expressivity of {_COLOR_SPACE.__name__} color space."
    return choices(_COLOR_SPACE, k=num_colors)


def draw(image: ImageAnnotation, label2color=None):
    if not isinstance(image, Composite):
        if not isinstance(image, (EmptyImage, Image)):
            image = Composite([image, EmptyImage(image.img_w, image.img_h)])
        else:
            image = Composite([image])

    assert image.num_image == 1, "At most one image is allowed for visualization."
    labels = image.unique_labels

    if label2color is None:
        label2color = dict(zip(labels, generate_colors(len(labels))))

    masks = [a for a in image.annotations if isinstance(a, Mask)]

    fig, axes = plt.subplots(1, len(masks) + 1)
    main_ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
    draw_image(main_ax, image.image)

    mask_count = 0
    for a in image.annotations:
        if isinstance(a, (Image, EmptyImage)):
            continue
        elif isinstance(a, Point):
            draw_point(main_ax, a, label2color)
        elif isinstance(a, PointArray):
            for p in a:
                draw_point(main_ax, p, label2color)
        elif isinstance(a, Box):
            draw_box(main_ax, a, label2color)
        elif isinstance(a, BoxArray):
            for b in a:
                draw_box(main_ax, b, label2color)
        elif isinstance(a, OrientedBox):
            draw_oriented_box(main_ax, a, label2color)
        elif isinstance(a, OrientedBoxArray):
            for b in a:
                draw_oriented_box(main_ax, b, label2color)
        elif isinstance(a, Polygon):
            draw_polygon(main_ax, a, label2color)
        else:
            assert isinstance(a, Mask), f"Unknown annotation type {type(a)}."
            ax = axes[1+mask_count]
            mask_count += 1
            draw_mask(ax, a, label2color)

    legend_elements = [
        lines.Line2D(
            [0], [0], marker='o', color=label2color[label], markerfacecolor=label2color[label],
            label=label, markersize=_POINT_RADIUS * 2
        ) for label in labels
    ]
    main_ax.legend(
        bbox_to_anchor=(0.5, -0.15), handles=legend_elements, loc='lower center',
        ncol=len(legend_elements)
    )

    title = ",".join([Label.hashable2str(label) for label in _get_values_of_interest(image.image.label)])
    main_ax.set_title(title)

    return fig
