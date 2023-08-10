This example demonstrates and explains how to use `image_transform` to train PyTorch object detectors (Faster-RCNN implemented by `torchvision`) with a dataset in COCO instance annotation format.

This project contains (for more details, refer the inline comments):
* `data.py`: define class `COCODetection`, a subclass of `torch.utils.Dataset` that:
    * Reads COCO object detection images and annotations in `__init__`.
    * Accesses and transforms the annotations, output `ImageAnnotation` and `Transform` in `__getitem__`. 
    * Provides `collate` for `torch.utils.Dataloader` mini-batching.
    * `self.scope` denoting the class names parsed from COCO annotation.
    * `model_minibatch_output_to_predictions()` to conver model output (`torch.Tensor`) to managible prediction (`BoxArray` with `ScopedWithConfidence`).
* `train_test.py`: provides `train()` and `validate()` that run the training and validation routines using utilities from `data.py`, `image_transform.metric` and `image_transform.transform`.
* `config.py`: definition of training transformations (with data augmentation), test transformations, mini-batch transformations and test post-processing transformations. These configurations are instantiated by `utils.parse_config()`.
* `utils.py`: `parse_config()`.
* `main.py`: start the training or validation process.

If you want to train your own model and data on the basis of this project, you can keep most of the code unchanged and only modify:
* `config.py`: define your transformations.
* `train_test.py`:
    * line 26: set your optimizer and learning rate scheduler;
    * line 50: convert the input to fit your model;
    * line 51-59: compute loss, update model parameters, log iteration information etc.;
    * line 61-63: validation;
    * line 72: choose your evaluation metric;
    * line 79: same as line 50;
    * line 95-100: convert your model output to bounding box predictions.
* `main.py`: `main()` prepare your dataset and model.
