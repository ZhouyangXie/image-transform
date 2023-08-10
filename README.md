### What's this

This package consists of useful utilities for image/image-based annotations(bounding boxes, oriented bounding boxes, points, masks, polygons etc.) such as transformations (padding, cropping, rotation, rescaling, resizing, flipping, transposition etc. and their automatically generated inverse), serialization from/to common dataset formats (Pascal VOC, COCO, Labelme etc.), evaluation metrics (classification, detection, segmentation etc.), and visualization.

### How to install

Clone this repo or unzip source code zip and execute this in root dir:
> pip install .

### How to use

Documents, in-code comments, and more examples are coming in the near future. For now, `examples` for a quick glimpse.

### Dependencies

numpy, opencv-python, matplotlib, Pillow

### TODO

Feature:

* Serialization to VOC/COCO/Labelme/Label Studio format.

* Evaluation metrics: more metrics.

* More transformations: Mosaic/CutOut/CutMix/AugMix/ColorJittering etc.

* GridCrop(MinibatchTransform): crop the whole image as a grid of crops, as the inverse, gather the annotations to a whole image.

* How to make use of existing libraries such as albumentations, to be compatible with more image formats and more transformations?

Project Quality:

* Fix error types. Runtime/Type/Attribute/Index Error or some custom Exception.

* Documents. Consider auto documenting tool like Sphinx.

* More comments/docstrings/type annotations/assert messages. In the comments, use `` to wrap the function/variable/type names.

* More tests. Make current tests more accurate.

* More examples/tutorials that cover model training/validation/inference on image classification/detection/segmentation/pose-estimation/image-synthesis etc.
