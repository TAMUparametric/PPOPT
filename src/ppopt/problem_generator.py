from typing import Optional

import numpy

from .mplp_program import MPLP_Program
from .mpqp_program import MPQP_Program


def generate_mplp(x: int = 2, t: int = 2, m: int = 10, seed: Optional[int] = None) -> MPLP_Program:
    """
    Generates a random mpLP problem with of the following characteristics

    :param x: number of parameters
    :param t: number of uncertain variables
    :param m: number of constraints
    :param seed: random seed
    :return: A random mpLP of the specified type
    """

    mpqp = generate_mpqp(x, t, m, seed)

    return MPLP_Program(mpqp.A, mpqp.b, mpqp.c, mpqp.H, mpqp.A_t, mpqp.b_t, mpqp.F)


def generate_mpqp(x: int = 2, t: int = 2, m: int = 10, seed: Optional[int] = None) -> MPQP_Program:
    """
    Generates a random mpQP problem with of the following characteristics

    :param x: number of x dimensions
    :param t: number of theta dimensions
    :param m: number of constraints
    :param seed: random seed
    :return: A random mpQP problem of the specified type
    """

    prng = numpy.random.default_rng(seed)

    Q = prng.random((x, x))
    Q = Q.T @ Q + numpy.eye(x)

    rand = lambda: prng.random(1)

    RangeValue = numpy.round(20 * rand() + 5)
    XBorder = numpy.round(8 * rand() + 1) / 10
    XShift = numpy.round(8 * rand() + 1) / 10
    TBorder = numpy.round(8 * rand() + 1) / 10
    TShift = numpy.round(8 * rand() + 1) / 10

    c = (prng.random((x, 1)) - .5) / rand()

    eigen_values = numpy.linalg.eigvals(Q)
    Range = RangeValue * (max(eigen_values) - min(eigen_values))

    A = numpy.zeros((m, x))
    F = numpy.zeros((m, t))

    for i in range(m):
        const = False
        while not const:
            guess = prng.random(x)
            idx = guess >= XBorder
            A[i][idx] = numpy.floor((prng.random(sum(idx)) - XShift) * Range)

            if any(A[i] != 0):
                const = True
        guess = prng.random(t)
        idx = guess >= TBorder
        F[i][idx] = numpy.floor((prng.random(sum(idx)) - TShift) * Range)

    A = numpy.block([[A], [numpy.eye(x)], [-numpy.eye(x)]])
    F = numpy.block([[F], [numpy.zeros((2 * x, t))]])

    b = numpy.block([[prng.random((m, 1)) / prng.random(1)], [10 ** 7 * numpy.ones((2 * x, 1))]])
    A_t = numpy.block([[numpy.eye(t)], [-numpy.eye(t)]])
    b_t = Range * numpy.ones((2 * t, 1))

    H = numpy.zeros((F.shape[1], Q.shape[0])).T
    return MPQP_Program(A, b, c, H, Q, A_t, b_t, F)
