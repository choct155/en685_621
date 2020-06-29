import numpy as np
from typing import Sequence
from algorithms.data_structures.Vector import Vector
from algorithms.data_structures.Matrix import Matrix

np_mat_in: np.array = np.arange(20).reshape(4,5)
vec_mat_in: Sequence[Vector] = [Vector(list(range(4))) for i in range(5)]

np_mat: Matrix = Matrix(np_mat_in)
vec_mat: Matrix = Matrix(vec_mat_in)

def test_init() -> None:
    print(np_mat)
    assert np_mat.data.all() == np_mat_in.all()