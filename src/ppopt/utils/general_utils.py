import os
from typing import Iterable, List, Union

import numpy


def make_column(x: Union[List, numpy.ndarray]) -> numpy.ndarray:
    """
    Makes x into a column vector

    :param x: a list or a numpy array
    :return: a numpy array that is a column vector
    """
    if isinstance(x, numpy.ndarray):
        return x.reshape(x.size, 1)
    return (numpy.array(x)).reshape(len(x), 1)


def make_row(x: Union[List, numpy.ndarray]) -> numpy.ndarray:
    """
    Makes x into a row vector

    :param x: a list or a numpy array
    :return: a numpy array that is a row column
    """
    if isinstance(x, numpy.ndarray):
        return x.reshape(1, x.size)
    return (numpy.array(x)).reshape(1, len(x))


def select_not_in_list(A: numpy.ndarray, list_: Iterable[int]) -> numpy.ndarray:
    """
    Filters a numpy array to select all rows that are not in a list

    :param A: a numpy array
    :param list_: a list of indices that you want to remove
    :return: return a numpy array of A[not in list_]
    """
    return A[[i for i in range(A.shape[0]) if i not in list_]]


def render_number(x, trade_off=1e-4) -> str:
    if isinstance(x, str):
        return x

    if abs(x) < 10 ** -14:
        return "0"
    elif abs(x) > trade_off:
        return f"{float(x):.4}"
    else:
        base_10 = numpy.log10(abs(x))
        exponent = 10 ** base_10
        return f"{x / exponent:.4} " + "10^{" + f"{exponent}" + "}"


def latex_matrix(A: Union[List[str], numpy.ndarray]) -> str:
    """
    Creates a latex string for a given numpy array

    :param A: A numpy array
    :return: A latex string for the matrix A
    """

    # beginning and ending of a matrix in latex
    start = "\\left[\\begin{matrix}"
    end = "\\end{matrix}\\right]"

    # generate an empty list of rows
    rows = []

    # if A is a matrix then make a matrix like object
    if isinstance(A, numpy.ndarray):
        for i in range(A.shape[0]):
            rows.append(" & ".join([render_number(j) for j in A[i]]))
        return start + "\\\\".join(rows) + end

    # default lists as column matrix
    if isinstance(A, list):
        return start + "\\\\".join([render_number(x) for x in A]) + end

    raise TypeError(f"When attempting to generate the latex rep of an object A, unsupported type {type(A)} was passed")


def remove_size_zero_matrices(list_matrices: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """
    Removes size zero matrices from a list

    :param list_matrices: A list of numpy arrays
    :return: returns all matrices from the list that do not have a dimension of 0 in any index
    """
    return [i for i in list_matrices if i.shape[0] > 0 and i.shape[1] > 0]


def num_cpu_cores():
    """
    Finds the number of allocated cores,with different behavior in windows and linux.

    In Windows, returns number of physical cpu cores

    In Linux, returns number of available cores for processing (this is for running on cluster or managed environment)

    :return: number of cores
    """

    cores = os.cpu_count()

    # noinspection SpellCheckingInspection
    if 'sched_getaffinity' in dir(os):
        cores = len(os.sched_getaffinity(0))

    return cores


def ppopt_block(mat_list):
    """
    This is an internal utility function that was created for internal use only for performance reasons. This is a
    replacement of ``numpy.block`` for performance sensitive sections of the codebase. This is approximately 3x faster
    for the matrices that are typically used here.

    :param mat_list: a list of matrices to concatenate in the same format as  ``numpy.block``
    :return: the concatenated matrix
    """

    # if the matrix list is of the form [A, F] transform it to [[A, F]] to simplify downstream logic
    if not isinstance(mat_list[0], list):
        mat_list = [mat_list]

    # find the size of the output matrix on the assumption that everything is properly sized
    x_size = sum(el.shape[1] for el in mat_list[0])
    y_size = sum(el[0].shape[0] for el in mat_list)

    # create the output buffer
    output_data = numpy.zeros((y_size, x_size))

    # set initial coordinates to start placing matrices
    x_cursor = 0
    y_cursor = 0

    # loop over all the matrix rows in the matrix list [[A, B, C], [D, E, F], ..., [Q, W, P]]
    for mat_row in mat_list:
        y_offset = 0

        # write out the matrix row [..., [A, B, ...,  Z], ....] into the row of the output buffer
        for matrix_ in mat_row:
            shape_ = matrix_.shape
            output_data[y_cursor: y_cursor + shape_[0], x_cursor: x_cursor + shape_[1]] = matrix_
            x_cursor += shape_[1]
            y_offset = shape_[0]

        # we are done with this row, move to the next row and reset the x coordinate
        y_cursor += y_offset
        x_cursor = 0

    # return the output buffer
    return output_data
