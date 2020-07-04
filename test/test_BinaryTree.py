import pytest
import numpy as np
from algorithms.data_structures.BinaryTree import EmptyNode, NonEmptyNode
from typing import List
from numbers import Rational

known_data: List[int] = [1,3,2,5,23,6]
sorted_data: List[int] = [1,2,3,5,6,23]

def test_append() -> None:
    one: NonEmptyNode = EmptyNode().append(1)
    three: NonEmptyNode = one.append(0).append(2)

    assert one.value == 1
    assert one.left.value == None
    assert one.right.value == None

    assert three.value == 1
    assert three.left.value == 0
    assert three.left.left.value == None
    assert three.left.right.value == None
    assert three.right.value == 2
    assert three.right.left.value == None
    assert three.right.right.value == None

def test_init_tree() -> None:
    truth: NonEmptyNode = (
        EmptyNode().append(1)
            .append(3)
            .append(2)
            .append(5)
            .append(23)
            .append(6)
    )
    assert NonEmptyNode.init_tree(known_data) == truth

def test_sorted_values() -> None:
    test: NonEmptyNode = NonEmptyNode.init_tree(known_data)
    assert test.sorted_values() == sorted_data