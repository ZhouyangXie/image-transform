import json
from math import pi

import matplotlib
import matplotlib.pyplot as plt

from image_transform.annotation.serialization import from_labelme
from image_transform.transform import Pad, Crop, CentralCrop, Rotate, RotateRightAngle, Resize,\
    Rescale, Transpose, HorizontalFlip, VerticalFlip, GaussianBlur, SequentialTransforms, PadTo
from image_transform.draw import draw, generate_colors


def main():
    # suppress interactive windows
    matplotlib.use('agg')

    with open('assets/labelme_example.json', 'r') as f:
        annotations = json.load(f)
    anno = from_labelme(annotations)

    # generate colors for the labels
    labels = anno.unique_labels
    colors = generate_colors(len(labels))
    label2color = dict((label, color) for label, color in zip(labels, colors))

    fig = draw(anno, label2color)
    fig.savefig('original.png')

    # initialize some transformations
    transforms = [
        # crop the image by a rectangle area (xmin, xmax, ymin, ymax)
        # the xmax-th column and the ymax-th row are included in the crop
        Crop(xmin=10, xmax=anno.img_w-21, ymin=20, ymax=anno.img_h-31),
        # pad the four sides of the image
        Pad(up=10, down=20, left=10, right=20),
        # flip the image veritically
        VerticalFlip(),
        # flip the image horizontally
        HorizontalFlip(),
        # rotate the image by 30°
        # note that the zero-point is the upper left corner of the image,
        # the x-axis is left-to-right, the y-axis is up-to-down,
        # the 0° is the right-side direction, and the rotation is performed clockwise
        Rotate(pi/6),
        # rotate the image by 90°
        # the only difference between rotate and rotate_right_angle is that when the angle
        # is the multiple of 90°, the rotated image is still a horizontal rectangle and no
        # interpolation is needed
        RotateRightAngle(90),
        # resize the image to 1.5 x width and 1.5 x height
        Rescale(1.5, 1.5),
        # resize the image to 2 x width and 2 x width
        Resize(anno.img_w * 2, anno.img_h * 2),
        # transpose the image(exchange the x- and y- axes)
        Transpose(),
        # perform two transformations sequentially, first symmetrically pad the image to
        # w=250, h=300, second crop the rectangle w=200, h=270 and the image center
        SequentialTransforms(transforms=[PadTo(250, 300), CentralCrop(220, 270)]),
        # Gaussian blurring the image. Other annotations are not affected.
        GaussianBlur(5)
    ]

    for transform in transforms:
        # transform the image
        transformed_anno = transform.transform(anno)
        # visualize the transformed image
        fig = draw(transformed_anno, label2color)
        fig.savefig(f'example_{type(transform).__name__}_transformed.png')
        plt.close(fig)

        # get the inverse transformation
        inverse_transform = transform.get_inverse()
        # perform the inverse transformation
        inversed_anno = inverse_transform.transform(transformed_anno)
        # visualize the inversely transformed versino of the transformed image
        # in most cases, it should resemble the original image and annotations
        fig = draw(inversed_anno, label2color)
        fig.savefig(f'example_{type(transform).__name__}_inversed.png')
        plt.close(fig)


if __name__ == "__main__":
    main()
