import numpy as np
import pytest

from ..image import EmptyImage
from ..label import Empty, ArbitraryHashable, Scoped, MultipleScoped,\
    ScopedWithConfidence, ProbabilisticMultipleScoped
from ..composite import Composite


def test_label():
    label = Empty()
    assert str(label) == "None"
    assert label == Empty()
    label = ArbitraryHashable("a")
    assert label == ArbitraryHashable("a")
    assert label != ArbitraryHashable("b")
    assert str(ArbitraryHashable("a")) == "a"
    assert str(ArbitraryHashable(1)) == str(1)

    class T:
        def __hash__(self) -> int:
            return 114514

        def __repr__(self) -> str:
            return str(hash(self))
    assert str(ArbitraryHashable(T())) == str(114514)

    class MyScoped(Scoped):
        scope = ["a", 1515, 3.14]

    with pytest.raises(AssertionError):
        _ = MyScoped(890)

    label = MyScoped(1515)
    assert label == MyScoped(1515)
    assert str(label) == str(1515)
    arr = np.array(2, dtype=np.int64)
    label = MyScoped.from_numpy(arr)
    assert label.value == 3.14
    assert np.all(label.to_numpy() == arr)
    assert label.to_multiple_scoped().values == [label.value]

    class MyScopedWithConfidence(ScopedWithConfidence):
        scope = ["a", 1919, 810]

    label = MyScopedWithConfidence("a", 0.1)
    assert str(label) == "a:0.10"
    index, confidence = label.to_numpy()
    assert np.isscalar(index) and index.dtype == np.int64 and index.item() == 0
    assert np.isscalar(confidence) and confidence.dtype == np.float32 and np.allclose(confidence.item(), 0.1)
    assert MyScopedWithConfidence.from_numpy((index, confidence)) == label

    class MyMultipleScoped(MultipleScoped):
        scope = ["b", None, False]

    label = MyMultipleScoped(["b", False])
    assert label == MyMultipleScoped([False, "b"])
    assert str(label) in ("{\'b\', False}", "{False, \'b\'}")
    assert np.allclose(label.to_numpy(), np.array([1, 0, 1], dtype=np.float32))
    l1 = MyMultipleScoped.from_numpy(np.array([0, -1, 2]))
    assert l1 == MyMultipleScoped([False, None])

    class MyProbabilisticMultipleScoped(ProbabilisticMultipleScoped):
        scope = MyScopedWithConfidence.scope

    label = MyProbabilisticMultipleScoped([2., 3., 5.])
    assert label == MyProbabilisticMultipleScoped([1., 1.5, 2.5])
    assert str(label) == "a:0.20,1919:0.30,810:0.50"
    assert MyProbabilisticMultipleScoped.from_numpy(label.to_numpy()) == label

    assert MyScoped(1515).to_multiple_scoped().values == [1515]
    label = MyScoped(1515).to_probabilistic_multiple_scoped()
    assert np.allclose(label.probs, np.array([0, 1, 0], np.float32))
    label = MyScopedWithConfidence(1919, 0.7).to_probabilistic_multiple_scoped(MyProbabilisticMultipleScoped)
    assert isinstance(label, MyProbabilisticMultipleScoped)
    assert np.allclose(label.probs, np.array([0.15, 0.7, 0.15], np.float32))
    assert MyMultipleScoped([None]).to_scoped().value is None
    label = MyProbabilisticMultipleScoped([0.8, 0.15, 0.05])
    assert label.to_scoped().value == 'a'
    assert label.to_scoped_with_confidence().value == 'a'
    assert np.allclose(label.to_scoped_with_confidence().confidence, .8)
    assert set(label.to_multiple_scoped(threshold=0.1).values) == {"a", 1919}

    composite = Composite([
        EmptyImage(40, 40, label=_label) for _label in
        [
            Empty(), ArbitraryHashable("arb"), MyScoped(1515), MyMultipleScoped((None, False)),
            MyScopedWithConfidence(1919, 0.1), MyProbabilisticMultipleScoped([.1, .1, .1])
        ]
    ])
    assert composite.unique_labels == set(("arb", 1515, None, False, 1919, "a", 1919, 810))
