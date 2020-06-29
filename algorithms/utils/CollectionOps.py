import numpy as np
from typing import TypeVar, Generic, Callable, Sequence

A = TypeVar('A')
B = TypeVar('B')

# TODO: Need to implement tail recurions: https://towardsdatascience.com/python-stack-frames-and-tail-call-optimization-4d0ea55b0542

def foldSeq(seq: Sequence[A], zero: B, func: Callable[[B, A], B]) -> B:
    out: B = zero
    for elem in seq:
        out: B = func(out, elem) 
        # Swallowing the update of `out` because it works as intended and is contained in scope
    return out
