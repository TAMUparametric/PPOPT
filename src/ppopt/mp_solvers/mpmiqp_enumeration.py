from typing import Union

from .mitree import MITree
from .solve_mpqp import mpqp_algorithm, solve_mpqp
from ..mpmilp_program import MPMILP_Program
from ..mpqp_program import MPQP_Program, MPLP_Program
from ..solution import Solution
from ..utils.general_utils import num_cpu_cores
from pathos.multiprocessing import ProcessingPool as Pool


def solve_mpmiqp_enumeration(program: MPMILP_Program, num_cores: int = -1,
                             cont_algorithm: mpqp_algorithm = mpqp_algorithm.combinatorial) -> Solution:
    """

    :param program: An mpQP/mpLP of a problem with the binary variables withot added constraints for the binary variables
    :param bin_variable_indices: the indices of the binary variables
    :param num_cores: the number of cores to use in this calculation to solve the mpLP/mpQPs
    :param cont_algorithm: the algorithm to solve the mpLP/mpQP algorithms
    :return: a solution that containts
    """
    # if core count is unspecified use all avalible cores
    if num_cores == -1:
        num_cores = num_cpu_cores()

    # generate problem tree
    tree = MITree(program, depth=0)

    feasible_combinations = [leaf_nodes.fixed_bins for leaf_nodes in tree.get_full_leafs()]
    problems = [program.generate_substituted_problem(fixed_bins) for fixed_bins in feasible_combinations]
    pool = Pool(num_cores)
    print(len(problems))
    sols = list(map(lambda x: solve_mpqp(x, cont_algorithm), problems))

    # add the fixed binary values to the critical regions
    region_list = []
    for index, sol in enumerate(sols):
        for i in range(len(sol.critical_regions)):
            sol.critical_regions[i].y_fixation = feasible_combinations[index]
            sol.critical_regions[i].y_indices = program.binary_indices
            sol.critical_regions[i].x_indices = program.cont_indices
        region_list.append(sol.critical_regions)

    #add the fixed binaries and indices to the critical regions

    # this has the possibility for overlapping critical regions so we set the overlapping flag
    enum_sol = Solution(program, [item for sublist in region_list for item in sublist], is_overlapping=True)

    return enum_sol
