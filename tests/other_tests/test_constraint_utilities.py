import numpy
from src.ppopt.utils.constraint_utilities import scale_constraint,remove_zero_rows,row_equality,remove_duplicate_rows,is_full_rank,cheap_remove_redundant_constraints,process_region_constraints, remove_strongly_redundant_constraints
from src.ppopt.utils.general_utils import make_row
import pytest


def test_constraint_norm_1():
    A = numpy.random.random((10, 10))
    b = numpy.random.random((10, 1))

    [As, _] = scale_constraint(A, b)

    results = numpy.linalg.norm(As, axis=1)
    assert numpy.allclose(numpy.ones(10), results)


def test_constraint_norm_2():
    A = -numpy.random.random((10, 10))
    b = numpy.random.random((10, 1))
    [_, _] = scale_constraint(A, b)

def test_scale_constraint():
    A = 2 * numpy.eye(3)
    b = numpy.ones(3)

    A, b = scale_constraint(A, b)

    assert numpy.allclose(A, numpy.eye(3))
    assert numpy.allclose(b, .5 * numpy.ones(3))


def test_remove_zero_rows():
    A = numpy.random.random((10, 10))
    b = numpy.random.random((10, 1))
    A[3] = 0
    A[7] = 0
    index = [0, 1, 2, 4, 5, 6, 8, 9]

    [A_, b_] = remove_zero_rows(A, b)
    assert numpy.allclose(A_, A[index])
    assert numpy.allclose(b_, b[index])
    assert A_.shape == A[index].shape
    assert b_.shape == b[index].shape


def test_row_equality_1():
    a = numpy.array([1, 2, 4])
    b = numpy.array([1, 2, 3])

    assert not row_equality(a, b)


def test_row_equality_2():
    a = numpy.array([1, 2, 3])
    b = numpy.array([1, 2, 3])

    assert row_equality(a, b)


def test_remove_duplicate_rows():
    A = numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 1, 1]])
    b = numpy.array([[1], [2], [1], [1]])
    [A, b] = remove_duplicate_rows(A, b)

    assert A.shape == (3, 3)
    assert b.shape == (3, 1)


def test_is_full_rank_1():
    A = numpy.eye(5)
    assert is_full_rank(A)


def test_is_full_rank_2():
    A = numpy.array([[1, 2, 3], [1, 0, 3]])
    assert is_full_rank(A)


def test_is_full_rank_3():
    A = numpy.eye(10)
    A[-1, -1] = 0
    assert not is_full_rank(A)


def test_is_full_rank_4():
    A = numpy.eye(4)
    assert is_full_rank(A, [1, 2, 3])


def test_is_full_rank_5():
    A = numpy.array([[1, 0], [1, 0], [0, 1]])
    assert not is_full_rank(A)
    assert not is_full_rank(A, [0, 1])
    assert is_full_rank(A, [1, 2])


def test_is_full_rank_6():
    A = numpy.eye(2)
    assert is_full_rank(A, [])


def test_remove_redundant_constraints():
    A = numpy.array([[-1, 0], [0, -1], [-1, 1], [-1, 1]])
    b = numpy.array([[0], [0], [1], [20]])

    # [As, bs] = process_region_constraints(A, b)
    As, bs = cheap_remove_redundant_constraints(A, b)
    As, bs = remove_strongly_redundant_constraints(As, bs)
    # As, bs = facet_ball_elimination(As, bs)

    A_ss, b_ss = scale_constraint(A, b)

    assert numpy.allclose(As, A_ss[[0, 1, 2]])
    assert numpy.allclose(bs, b_ss[[0, 1, 2]])


def test_process_region_constraints():
    A = numpy.block([[numpy.eye(3)], [-numpy.eye(3)], [make_row([1, 1, 1])]])

    b = numpy.block([[numpy.ones((3, 1))], [numpy.zeros((3, 1))], [numpy.array([[1]])]])

    [A, b] = process_region_constraints(A, b)

    assert A.shape == (4, 3)
    assert b.shape == (4, 1)


@pytest.mark.skip(reason="I am scaling the matrix array, expected output has changed")
def test_facet_ball_elimination():
    A = numpy.block([[numpy.eye(2)], [-numpy.eye(2)]])
    b = numpy.array([[1], [1], [0], [0]])

    A_t = numpy.block([[numpy.eye(2)], [-numpy.eye(2)], [numpy.array([[1, 1]])]])
    b_t = numpy.array([[2], [2], [0], [0], [1]])

    A_r = numpy.block([[A], [A_t]])
    b_r = numpy.block([[b], [b_t]])

    [_, _] = process_region_constraints(A_r, b_r)

