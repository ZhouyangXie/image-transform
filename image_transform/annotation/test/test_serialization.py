from ..serialization import from_labelme, to_labelme, from_voc, make_coco_object_detection_accessor


def test_io():
    # better check by eyes the loaded or dumped annotation is correct
    composite = from_labelme("assets/labelme_example.json")
    _ = to_labelme(composite, dst=None)

    _ = from_voc("assets/voc/Annotations/000001.xml", "assets/voc/JPEGImages")

    annotations = make_coco_object_detection_accessor(
        "assets/coco/instances_sample2023.json",
        "assets/coco"
    )
    assert len(annotations) == 10
    _ = annotations[0]
