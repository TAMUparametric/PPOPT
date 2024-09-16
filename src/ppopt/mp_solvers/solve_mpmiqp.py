from enum import Enum

from ..mpmilp_program import MPMILP_Program
from ..solution import Solution
from .mpmiqp_enumeration import solve_mpmiqp_enumeration
from .solve_mpqp import mpqp_algorithm, solve_mpqp


class mpmiqp_algorithm(Enum):
    """
    Enum that selects the mpmiqp algorithm to be used

    This is done by passing the argument mpmiqp_algorithm.algorithm

    This is typically combined in conjunction with a mpqp_algorithm to solve sub-problems when they arise
    """
    enumerate = 'enumerate'

    def __str__(self):
        return self.name

    @staticmethod
    def all_algos():
        output = ''
        for algo in mpmiqp_algorithm:
            output += f'mpmiqp_algorithm.{algo}\n'
        return output

def solve_mpmiqp(problem: MPMILP_Program, mpmiqp_algo: mpmiqp_algorithm = mpmiqp_algorithm.enumerate,
                 cont_algo: mpqp_algorithm = mpqp_algorithm.combinatorial, num_cores=-1) -> Solution:

    # the case of a continuous problem just solve it and return
    if len(problem.binary_indices) == 0:
        print("The problem does not have any binary variables, solving as a continuous problem instead.")
        # noinspection PyTypeChecker
        return solve_mpqp(problem, cont_algo)

    if not isinstance(mpmiqp_algo, mpmiqp_algorithm):
        raise TypeError(
            f"You must pass an algorithm from mpmiqp_algorithm as the continuous algorithm. These can be found by "
            f"importing the following \n\nfrom ppopt.mp_solvers.solve_mpmiqp import mpmiqp_algorithm\n\nWith the "
            f"following choices\n{mpmiqp_algorithm.all_algos()}")

    # listing of all available algorithms
    if mpmiqp_algo == mpmiqp_algorithm.enumerate:
        return solve_mpmiqp_enumeration(problem, num_cores, cont_algo)

    # this shouldn't happen but if you ask for an unavailable algorithm you get an emtpy solution
    return Solution(problem, [])
