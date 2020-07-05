import numpy as np
from typing import TypeVar, Generic, Callable, Sequence, List, Tuple

A = TypeVar('A')
B = TypeVar('B')

# TODO: Need to implement tail recurions: https://towardsdatascience.com/python-stack-frames-and-tail-call-optimization-4d0ea55b0542

def foldSeq(seq: Sequence[A], zero: B, func: Callable[[B, A], B]) -> B:
    out: B = zero
    for elem in seq:
        out: B = func(out, elem) #type: ignore
        # Swallowing the update of `out` because it works as intended and is contained in scope
    return out

def mapList(seq: List[A], func: Callable[[A], B]) -> List[B]:
    # Not sure how to represent an empty sequence, so list it is
    f: Callable[[List[B], A], List[B]] = lambda out, next: out + [func(next)]
    return foldSeq(seq, [], f)

def filterList(seq: List[A], pred: Callable[[A], bool]) -> List[A]:
    f: Callable[[List[A], A], List[A]] = lambda out, next: out + [next] if pred(next) else out
    return foldSeq(seq, [], f)

def splitList(seq: List[A], pred: Callable[[A], bool]) -> Tuple[List[A], List[A]]:
    """
    Splits the list based upon the filter condition. Failing elements are allocated
    to the list on the left (index position 0), and passing elements are allocated
    to the list on the right (index position 1).
    """
    def f(out: Tuple[List[A], List[A]], next: A) -> Tuple[List[A], List[A]]:
        old_fail, old_pass = out
        if pred(next):
            return (old_fail, old_pass + [next])
        else:
            return (old_fail + [next], old_pass)
    zero: Tuple[List[A], List[A]] = ([],[])
    return foldSeq(seq, zero, f)


def concatLists(this: List[A], that: List[A]) -> List[A]:
    return foldSeq(that, this, lambda out, next_val: out + [next_val])

def zipLists(this: List[A], that: List[A]) -> List[Tuple[A, A]]:
    """ 
    Zips two lists together based upon the shorter list so that 
    all tuples are populated. Avoids the need for an empty value
    of A. The lack of implicits is annoying.
    """
    index_list: List[A] = this if len(this) <= len(that) else that
    f: Callable[[List[Tuple[A, A]], int], List[Tuple[A, A]]] = lambda out, idx: out + [(this[idx], that[idx])]
    return foldSeq(range(len(index_list)), [], f)

def equalLists(this: List[A], that: List[A]) -> bool:
    if len(this) != len(that):
        return False
    else:
        zipped: List[Tuple[A, A]] = zipLists(this, that)
        f: Callable[[bool, Tuple[A, A]], bool] = lambda still_equal, next_pair: still_equal & (next_pair[0] == next_pair[1])
        return foldSeq(zipped, True, f)

def combineListElems(this: List[A], that: List[A], combine: Callable[[A, A], A]) -> List[A]:
    zipped: List[Tuple[A, A]] = zipLists(this, that)
    f: Callable[[List[A], Tuple[A, A]], List[A]] = lambda out, next_pair: out + [combine(next_pair[0], next_pair[1])]
    return foldSeq(zipped, [], f)


# Semantically familiar, but perhaps unnecessary
def head(seq: Sequence[A]) -> A:
    h, *t = seq
    return h

def tail(seq: Sequence[A]) -> List[A]:
    h, *t = seq
    return t