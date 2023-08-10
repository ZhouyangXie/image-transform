import numpy as np

from ...annotation import BoxArray, Scoped, ScopedWithConfidence
from ..detection import ClassAgnosticAccuracyWithConfidence, AccuracyWithConfidence


def test_empty_target_or_prediction():
    m = ClassAgnosticAccuracyWithConfidence()
    m.update(np.zeros((5, 0)), np.ones(5), 1)
    r = m.get_score()
    assert np.allclose(r['recall'], 1)
    assert np.allclose(r['average precision'], 0)
    assert np.allclose(r['count'], 0)
    m.reset()
    m.update(np.zeros((0, 5)), np.ones(0), 10)
    r = m.get_score()
    assert np.allclose(r['recall'], 0)
    assert np.allclose(r['average precision'], 0)
    assert np.allclose(r['count'], 5)


class _MyScoped(Scoped):
    scope = [0]


class _MyScopedWithConfidence(ScopedWithConfidence):
    scope = [0]


def _make_prediction_or_target(box, labels, confidence=None):
    if confidence is None:
        labels = [_MyScoped(int(label)) for label in labels]
    else:
        labels = [_MyScopedWithConfidence(int(label), c) for label, c in zip(labels, confidence)]

    box = np.array(box)
    xmin, ymin, xmax, ymax = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    assert len(box) == len(labels)
    return BoxArray(xmin, xmax, ymin, ymax, 1000, 1000, labels)


def test_detection_with_confidence_A():
    m = AccuracyWithConfidence(scope=[0, ])
    predictions = _make_prediction_or_target(
        box=[
            [0, 0, 30, 30],
            [0, 0, 12, 12],
            [0, 0, 9, 9],
            [0, 0, 2, 2],
            [0, 0, 999, 999],
            [9, 9, 19, 19],
        ],
        labels=[0, 0, 0, 0, 0, 0],
        confidence=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    )
    targets = _make_prediction_or_target(
        box=[
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [20, 20, 30, 30],
        ],
        labels=[0, 0, 0],
    )
    m.update([predictions], [targets])
    r = m.get_score()
    assert np.allclose(r['recalls'], np.array([2/3]))
    assert np.allclose(r['mean recall'], 2/3)
    assert np.allclose(r['average precisions'], np.array([5/18]))
    assert np.allclose(r['mean average precision'], 5/18)
    assert np.allclose(r['counts'], np.array([3]))


def test_detection_with_confidence_B():
    m = AccuracyWithConfidence(scope=[0, ], iou_thresholds=0.2)
    predictions = _make_prediction_or_target(
        box=[
            [1, 1, 3, 3],
            [-1, -1, 0., 0.],
            [-1, -1, 0., 0.],
            [0, 0, 1, 1],
            [0, 0, 100, 100],
        ],
        labels=[0, 0, 0, 0, 0],
        confidence=[0.4, 0.5, 0.6, 0.7, 0.8, ]
    )
    targets = _make_prediction_or_target(
        box=[
            [0, 0, 1, 1],
            [1, 1, 2, 2],
            [2, 2, 3, 3],
        ],
        labels=[0, 0, 0],
    )
    m.update([predictions], [targets])
    r = m.get_score()
    assert np.allclose(r['recalls'], np.array([1.]))
    assert np.allclose(r['mean recall'], 1.)
    assert np.allclose(r['average precisions'], np.array([13/30]))
    assert np.allclose(r['mean average precision'], 13/30)
    assert np.allclose(r['counts'], np.array([3]))


def test_detection_with_confidence_C():
    m = AccuracyWithConfidence(scope=[0, ], iou_thresholds=0.2499)
    predictions = _make_prediction_or_target(
        box=[
                [0, 0, 10, 10],
                [0., 0., 1., 1.],
                [-1, -1, 0., 0.],
                [1, 1, 2, 2],
                [1, 1, 3, 3],
        ],
        labels=[0, 0, 0, 0, 0],
        confidence=[0.9, 0.8, 0.7, 0.6, 0.5]
    )
    targets = _make_prediction_or_target(
        box=[
            [0, 0, 1, 1],
            [1, 1, 2, 2],
            [2, 2, 3, 3],
        ],
        labels=[0, 0, 0],
    )
    m.update([predictions], [targets])
    r = m.get_score()
    assert np.allclose(r['recalls'], np.array([1.]))
    assert np.allclose(r['mean recall'], 1.)
    assert np.allclose(r['average precisions'], np.array([.6]))
    assert np.allclose(r['mean average precision'], .6)
    assert np.allclose(r['counts'], np.array([3]))


def test_detection_with_confidence_D():
    m = AccuracyWithConfidence(scope=[0, ])
    predictions = _make_prediction_or_target(
        box=[
            [0, 0, 10, 10],
            [0., 0., 1., 1.],
            [-1, -1, 0., 0.],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 10, 10],
            [1, 1, 10, 10],
        ],
        labels=[0, 0, 0, 0, 0, 0, 0],
        confidence=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    )
    targets = _make_prediction_or_target(
        box=[
            [0, 0, 1, 1],
            [1, 1, 2, 2],
            [2, 2, 3, 3],
            [3, 3, 4, 4]
        ],
        labels=[0, 0, 0, 0],
    )
    m.update([predictions], [targets])
    r = m.get_score()
    assert np.allclose(r['recalls'], np.array([.5]))
    assert np.allclose(r['mean recall'], .5)
    assert np.allclose(r['average precisions'], np.array([0.25]))
    assert np.allclose(r['mean average precision'], 0.25)
    assert np.allclose(r['counts'], np.array([4]))
