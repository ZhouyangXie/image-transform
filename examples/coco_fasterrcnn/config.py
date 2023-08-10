transforms = dict(
    training_transform=[
        # keep boxes and images only
        dict(type="Filter", types=["Box", "BoxArray", "Image"]),
        # data augmentation
        dict(type="RandomVerticalFlip"),
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomTranspose"),
        dict(type="RandomRotateRightAngle"),
        # pad the image to the smallest multiple of 32
        dict(type="PadToMultiple", w_base=32, h_base=32),
        # if the image is gray-scale, convert to RGB
        dict(type="Gray2RGB"),
        # convert image 256 color (uint8) to 0~1 intensity (float32)
        dict(type="ToDataType", to_dtype='float32'),
        # Mean-std normalization is not included here because torchvision faster-rcnn does it inside the model.
        # Your model will probably need it outside. Add it after ToDataType as:
        # `dict(type="Normalize", mean=[...], std=[...])``
        # and do not forget test_transform.
    ],
    test_transform=[
        dict(type="Filter", types=["Box", "BoxArray", "Image"]),
        dict(type="PadToMultiple", w_base=32, h_base=32),
        dict(type="Gray2RGB"),
        dict(type="ToDataType", to_dtype='float32'),
    ],
    minibatch_transform=[
        # pad all images to the max width/height of this mini-batch
        dict(type="PadToMax"),
        # to np.ndarray
        dict(type="Stack", image_as_feature_map=True, box_format="xyxy")
    ],  # shared by training and testing
    test_post_processing=[
        # leave out all boxes with a ScopedWithConfidence label confidence < 0.2
        dict(type="ConfidenceThreshold", threshold=0.2),
        # class aware non maximum suppression on the boxes
        dict(type="NonMaximumSuppression", class_aware=True, iou_threshold=0.5)
    ]
)
