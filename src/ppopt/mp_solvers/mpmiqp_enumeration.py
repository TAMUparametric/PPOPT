from typing import Union

from .mitree import MITree
from .solve_mpqp import mpqp_algorithm, solve_mpqp
from ..mpqp_program import MPQP_Program, MPLP_Program
from ..solution import Solution
from ..utils.general_utils import num_cpu_cores
from pathos.multiprocessing import ProcessingPool as Pool


def solve_mpmiqp_enumeration(program: Union[MPQP_Program, MPLP_Program], bin_variable_indices: list = None,
                             num_cores: int = -1,
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
    tree = MITree(program, bin_indices=bin_variable_indices, depth=0)

    problems = [leaf_nodes.problem for leaf_nodes in tree.get_full_leafs()]
    pool = Pool(num_cores)

    sols = list(pool.map(lambda x: solve_mpqp(x, cont_algorithm), problems))

    # extract all critical region from the sub mpLPs
    region_list = [sol.critical_regions for sol in sols]

    # this has the possibility for overlapping critical regions so we set the overlapping flag
    enum_sol = Solution(program, [item for sublist in region_list for item in sublist], is_overlapping=True)

    return enum_sol
