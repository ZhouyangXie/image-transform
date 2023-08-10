from image_transform.transform import all_names, SequentialTransforms


examples_transforms = dict(
    training_transform=[
        dict(type="Filter", types=["Box", "Image"]),
        dict(type="RandomVerticalFlip"),
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomTranspose"),
        dict(type="RandomRotateRightAngle"),
        dict(type="Resize", dst_w=512, dst_h=512),
        dict(type="ToDataType", to_dtype='float32'),
        dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ],
    test_transform=[
        dict(type="Filter", types=["Box", "Image"]),
        dict(type="Resize", dst_w=512, dst_h=512),
        dict(type="ToDataType", to_dtype='float32'),
        dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ],
    minibatch_transform=dict(type="ToNumpy", image_as_feature_map=True, box_format="xyxy"),
)


def parse_config(transform):
    if isinstance(transform, dict):
        assert "type" in transform
        transform = transform.copy()
        transform_type_name = transform.pop("type")
        assert transform_type_name in all_names
        return all_names[transform_type_name](**transform)
    elif isinstance(transform, list):
        return SequentialTransforms([parse_config(t) for t in transform])

    else:
        raise ValueError


if __name__ == "__main__":
    training_tranform = parse_config(examples_transforms["training_transform"])
    test_transform = parse_config(examples_transforms["test_transform"])
    minibatch_transform = parse_config(examples_transforms["minibatch_transform"])
