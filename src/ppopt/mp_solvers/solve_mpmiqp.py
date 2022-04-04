from enum import Enum

from .solve_mpqp import mpqp_algorithm, solve_mpqp
from ..mpmilp_program import MPMILP_Program
from ..mpqp_program import MPQP_Program
from ..solution import Solution

from .mpmiqp_enumeration import solve_mpmiqp_enumeration


class mpmiqp_algorithm(Enum):
    """
    Enum that selects the mpmiqp algorithm to be used

    This is done by passing the argument mpmiqp_algorithm.algorithm

    This is typically combined in conjunction with a mpqp_algorithm to solve sub-problems when they arise
    """
    enumerate = 'enumerate'


def solve_mpmiqp(problem: MPMILP_Program, mpmiqp_algo: mpmiqp_algorithm = mpmiqp_algorithm.enumerate,
                 cont_algo: mpqp_algorithm = mpqp_algorithm.combinatorial, num_cores=-1) -> Solution:
    # the case of a continuous problem just solve it and return
    if len(problem.binary_indices) == 0:
        print("The PROBLEM DOES NOT HAVE ANY BINARY VARIABLES!!!")
        # noinspection PyTypeChecker
        return solve_mpqp(problem, cont_algo)

    # listing of all available algorithms
    if mpmiqp_algo == mpmiqp_algorithm.enumerate:
        return solve_mpmiqp_enumeration(problem, num_cores, cont_algo)

    # this shouldn't happen but if you ask for an unavailable algorithm you get an emtpy solution
    return Solution(problem, [])
