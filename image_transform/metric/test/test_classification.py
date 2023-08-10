import numpy as np
from pytest import raises

from ..classification import MulticlassAccuracy, MultilabelAccuracy


def test_multiclass_accurarcy():
    metric = MulticlassAccuracy(['A', 0, 3.14])

    with raises(AssertionError):
        metric.update(['A'], ['A', 0])

    with raises(ValueError):
        metric.update(['A'], ['B'])

    metric.update(
        [3.14, 0, 0],
        ['A', 'A', 'A']
    )
    metric.update(
        ['A', 0, 3.14],
        [0, 0, 0]
    )
    metric.update(
        [3.14, 3.14, 3.14],
        [3.14, 3.14, 3.14]
    )

    results = metric.get_score()
    assert np.allclose(results['recalls'], np.array([0., 1/3, 1.0]))
    assert np.allclose(results['mean recall'], 4/9)
    assert np.allclose(results['precisions'], np.array([0., 1/3, 3/5]))
    assert np.allclose(results['mean precision'], 14/45)
    assert np.allclose(results['f1scores'], np.array([0., 1/3, 3/4]))
    assert np.allclose(results['mean f1score'], 13/36)
    assert np.allclose(
        results['confusion'],
        np.array(
            [
                [0, 1, 0],
                [2, 1, 0],
                [1, 1, 3],
            ], dtype=int
        )
    )
    assert np.allclose(results["accuracy"], 4/9)
    assert np.allclose(results["count"], np.array([3, 3, 3]))


def test_multiclass_zero_instance_zero_prediction():
    metric = MulticlassAccuracy([0, 1])
    metric.update(
        [1, 1, 1],
        [0, 0, 0]
    )
    results = metric.get_score()
    assert np.allclose(results['recalls'], np.array([0., 1.]))
    assert np.allclose(results['mean recall'], 0.5)
    assert np.allclose(results['precisions'], np.array([1., 0.]))
    assert np.allclose(results['mean precision'], 0.5)
    assert np.allclose(results['f1scores'], np.array([0., 0.]))
    assert np.allclose(results['mean f1score'], 0.)
    assert np.allclose(
        results['confusion'],
        np.array(
            [
                [0, 0],
                [3, 0],
            ], dtype=int
        )
    )


def test_multilabel_accurarcy():
    metric = MultilabelAccuracy(['a', 'b', 'c'])

    metric.update(
        np.array([
            [True, False, False],
            [False, False, True],
            [False, False, True],
            [True, False, False],
        ], bool),
        np.array([
            [True, False, False],
            [True, False, False],
            [False, True, False],
            [False, True, False],
        ], bool),
    )

    results = metric.get_score()
    assert np.allclose(results['recalls'], np.array([1/2, 0., 1.0]))
    assert np.allclose(results['mean recall'], 1/2)
    assert np.allclose(results['precisions'], np.array([1/2, 1.0, 0.0]))
    assert np.allclose(results['mean precision'], 1/2)
    assert np.allclose(results['f1scores'], np.array([1/2, 0., 0.]))
    assert np.allclose(results['mean f1score'], 1/6)
    assert np.allclose(
        results['confusions'],
        np.array(
            [
                [
                    [1, 1],
                    [1, 1],
                ],
                [
                    [2, 2],
                    [0, 0],
                ],
                [
                    [2, 0],
                    [2, 0],
                ],
            ], dtype=int
        )
    )
    assert np.allclose(results["accuracies"], np.array([.5, .5, .5]))
    assert np.allclose(results["counts"], np.array([2, 2, 0]))
