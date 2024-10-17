from enum import Enum
import logging

from ..mpmilp_program import MPMILP_Program
from ..solution import Solution
from .mpmiqp_enumeration import solve_mpmiqp_enumeration
from .solve_mpqp import mpqp_algorithm, solve_mpqp

import numpy

from ..utils.region_overlap_utils import reduce_overlapping_critical_regions_1d

logger = logging.getLogger(__name__)

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
                 cont_algo: mpqp_algorithm = mpqp_algorithm.combinatorial, num_cores=-1,
                 reduce_overlap=True) -> Solution:
    # the case of a continuous problem just solve it and return
    if len(problem.binary_indices) == 0:
        logger.warning("The problem does not have any binary variables, solving as a continuous problem instead.")
        # noinspection PyTypeChecker
        return solve_mpqp(problem, cont_algo)

    if not isinstance(mpmiqp_algo, mpmiqp_algorithm):
        raise TypeError(
            f"You must pass an algorithm from mpmiqp_algorithm as the continuous algorithm. These can be found by "
            f"importing the following \n\nfrom ppopt.mp_solvers.solve_mpmiqp import mpmiqp_algorithm\n\nWith the "
            f"following choices\n{mpmiqp_algorithm.all_algos()}")

    cand_sol = Solution(problem, [])

    # listing of all available algorithms
    if mpmiqp_algo == mpmiqp_algorithm.enumerate:
        cand_sol = solve_mpmiqp_enumeration(problem, num_cores, cont_algo)

    # see if we can actually reduce the overlaps given current limitations
    sum_abs_H = numpy.sum(numpy.abs(problem.H[problem.cont_indices, :]))
    is_bilinear_terms: bool = not numpy.isclose(sum_abs_H, 0)

    # we currently only support for pMILP problems no H term
    if not (problem.num_t() > 1 or hasattr(problem, 'Q') or not reduce_overlap or is_bilinear_terms):
        # For 1D mpMILP case, we remove overlaps
        # In case of dual degeneracy we keep all solutions so in this case there could still be overlaps
        collected_regions, is_overlapping = reduce_overlapping_critical_regions_1d(problem, cand_sol.critical_regions)
        return Solution(problem, collected_regions, is_overlapping)

    return cand_sol
