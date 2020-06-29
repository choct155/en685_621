import pytest
from typing import Sequence, Callable
from algorithms.utils.CollectionOps import foldSeq


def test_foldSeq() -> None:
    data: Sequence[int] = list(range(1, 6))
    stringConcat: Callable[[int, int], str] = lambda x, y: str(x) + str(y)
    out: str  = foldSeq(data, "", stringConcat)
    assert out == "12345"