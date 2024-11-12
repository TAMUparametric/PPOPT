# this is the interface for the mpQP and mpLP problem solvers

from enum import Enum

import numpy

from ..mp_solvers import (
    mpqcqp_combinatorial,
)
from ..mplp_program import MPLP_Program
from ..mpqp_program import MPQP_Program
from ..mpqcqp_program import MPQCQP_Program
from ..solution import Solution


class mpqcqp_algorithm(Enum):
    """
    Enum that selects the mpqp algorithm to be used

    This is done by passing the argument mpqp_algorithm.algorithm
    """
    combinatorial = 'combinatorial'

    def __str__(self):
        return self.name

    @staticmethod
    def all_algos():
        output = ''
        for algo in mpqcqp_algorithm:
            output += f'mpqp_algorithm.{algo}\n'
        return output


def solve_mpqcqp(problem: MPQP_Program, algorithm: mpqcqp_algorithm = mpqcqp_algorithm.combinatorial) -> Solution:
    """
    Takes a mpqcqp programming problem and solves it in a specified manner. The default
     solve algorithm is the Combinatorial algorithm by Gupta. et al.

    :param problem: Multiparametric Program to be solved
    :param algorithm: Selects the algorithm to be used
    :return: the solution of the MPQCQP, returns an empty solution if there is not an implemented algorithm
    """

    if not isinstance(algorithm, mpqcqp_algorithm):
        raise TypeError(
            f"You must pass an algorithm from mpqp_algorithm as the continuous algorithm. These can be found by "
            f"importing the following \n\nfrom ppopt.mp_solvers.solve_mpqp import mpqp_algorithm\n\nWith the "
            f"following choices\n{mpqcqp_algorithm.all_algos()}")

    solution = Solution(problem, [])

    if algorithm is mpqcqp_algorithm.combinatorial:
        solution = mpqcqp_combinatorial.solve(problem)

    # check if there needs to be a flag thrown in the case of overlapping critical regions
    # happens if there are negative or zero eigen values for mpQP (kkt conditions can find a lot of saddle points)
    if isinstance(problem, MPQP_Program):
        if min(numpy.linalg.eigvalsh(problem.Q)) <= 0:
            solution.is_overlapping = True

    # in the case of degenerate problems there are overlapping critical regions, unless a check is performed to prove
    # no overlap it is generally safer to consider that the mpLP case is overlapping
    if isinstance(problem, MPLP_Program):
        solution.is_overlapping = True

    return filter_solution(solution)


def filter_solution(solution: Solution) -> Solution:
    """
    This is a placeholder function, in the future this will be used to process and operate on the solution before it
    is returned to the user.

    :param solution: a multi parametric solution

    :return: A processed solution
    """

    return solution
