from enum import Enum

from .solve_mpqp import mpqp_algorithm, solve_mpqp
from ..mpqp_program import MPQP_Program
from ..solution import Solution

from .mpmiqp_enumeration import solve_mpmiqp_enumeration

class mpmiqp_algorithm(Enum):
    """
    Enum that selects the mpmiqp algorithm to be used

    This is done by passing the argument mpmiqp_algorithm.algorithm

    This is typically combined in conjunction with an mpqp_algorithm to solve subproblems when they arise
    """
    enumerate = 'enumerate'


def solve_mpmiqp(problem: MPQP_Program, bin_indices:list = None, mpmiqp_algo: mpmiqp_algorithm = mpmiqp_algorithm.enumerate, cont_algo: mpqp_algorithm = mpqp_algorithm.combinatorial, num_cores = -1) -> Solution:

    # the case of a continuous problem just solve it and return
    if bin_indices is None:
        bin_indices = []
        return solve_mpqp(problem, cont_algo)

    # listing of all available algorithms
    if mpmiqp_algo == mpmiqp_algorithm.enumerate:
        return solve_mpmiqp_enumeration(problem, bin_indices, num_cores, cont_algo)

    # this shouldn't happen but if you ask for an unavalible algorithm you get an emtpy solution
    return Solution(problem, [])