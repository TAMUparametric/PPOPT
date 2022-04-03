from typing import Union

from .mitree import MITree
from .solve_mpqp import mpqp_algorithm, solve_mpqp
from ..mpmilp_program import MPMILP_Program
from ..mpqp_program import MPQP_Program, MPLP_Program
from ..solution import Solution
from ..utils.general_utils import num_cpu_cores
# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool


def solve_mpmiqp_enumeration(program: MPMILP_Program, num_cores: int = -1,
                             cont_algorithm: mpqp_algorithm = mpqp_algorithm.combinatorial) -> Solution:
    """
    The enumeration algorithm is based on the following approach

    1) Enumerating all feasible binary combinations
    2) Solving the resulting continuous mpQP/mpLP sub-problems for every feasible binary combination
    3) Merging all solutions together

    :param program: An mpQP/mpLP of a problem with the binary variables without added constraints for the binary variables
    :param num_cores: the number of cores to use in this calculation to solve the mpLP/mpQP sub-problems
    :param cont_algorithm: the algorithm to solve the mpLP/mpQP algorithms (might not be required)
    :return: a solution to the mpMILP/mpMIQP (might have overlapping critical regions depending on algorithm choice)
    """
    # if core count is unspecified use all available cores
    if num_cores == -1:
        num_cores = num_cpu_cores()

    # generate problem tree
    tree = MITree(program, depth=0)

    # grab all feasible binary combinations
    feasible_combinations = [leaf_nodes.fixed_bins for leaf_nodes in tree.get_full_leafs()]

    # generate all substituted problems from these binary combinations to make continuous sub-problems
    problems = [program.generate_substituted_problem(fixed_bins) for fixed_bins in feasible_combinations]

    # make a thread pool then solve all problems in parallel with the supplied continuous algorithm
    pool = Pool(num_cores)
    sols = list(pool.map(lambda x: solve_mpqp(x, cont_algorithm), problems))

    # add the fixed binary values to the critical regions
    region_list = []
    for index, sol in enumerate(sols):
        for i in range(len(sol.critical_regions)):
            # add the fixed binary combination, the binary indices and the continuous variable indices
            sol.critical_regions[i].y_fixation = feasible_combinations[index]
            sol.critical_regions[i].y_indices = program.binary_indices
            sol.critical_regions[i].x_indices = program.cont_indices
        region_list.append(sol.critical_regions)

    # this has the possibility for overlapping critical regions, so we set the overlapping flag
    enum_sol = Solution(program, [item for sublist in region_list for item in sublist], is_overlapping=True)

    return enum_sol
