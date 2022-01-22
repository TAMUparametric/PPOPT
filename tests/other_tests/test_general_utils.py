import numpy
from src.ppopt.utils.general_utils import make_column, make_row, select_not_in_list, remove_size_zero_matrices


def test_make_column_1():
    test_case = make_column([1, 1, 1, 1])
    correct_result = numpy.array([[1], [1], [1], [1]])

    assert numpy.allclose(correct_result, test_case)
    assert correct_result.shape == test_case.shape


def test_make_column_2():
    k = numpy.ones((2, 2))
    assert make_column(k).shape == (4, 1)


def test_make_column_3():
    k = numpy.ones((2,))
    assert make_column(k).shape == (2, 1)


def test_make_row_1():
    test_case = make_row([1, 1, 1, 1])
    correct_result = numpy.array([[1, 1, 1, 1]])

    assert numpy.allclose(correct_result, test_case)
    assert correct_result.shape == test_case.shape


def test_make_row_2():
    k = numpy.ones((2, 2))
    assert make_row(k).shape == (1, 4)


def test_make_row_3():
    k = numpy.ones((2,))
    assert make_row(k).shape == (1, 2)


def test_select_not_in_list_1():
    A = numpy.eye(5)
    B = select_not_in_list(A, [0])
    assert numpy.allclose(A[[1, 2, 3, 4]], B)


def test_select_not_in_list_2():
    A = numpy.eye(5)
    B = select_not_in_list(A, [0, 1, 2, 3, 4])
    assert B.size == 0


def test_remove_size_zero_matrices():
    A = [numpy.eye(0), numpy.eye(1), numpy.zeros((2, 0))]
    assert remove_size_zero_matrices(A) == [numpy.eye(1)]

