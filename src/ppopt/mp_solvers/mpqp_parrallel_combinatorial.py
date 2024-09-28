from src.ppopt.critical_region import CriticalRegion
import time
from random import shuffle
from typing import List, Set, Tuple, Optional

# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool

from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.general_utils import num_cpu_cores
from ..utils.mpqp_utils import gen_cr_from_active_set
from .solver_utils import CombinationTester, generate_children_sets


def full_process(program: MPQP_Program, active_set: List[int], murder_list, gen_children) -> Tuple[Optional[CriticalRegion], Set[Tuple[int,...]], List[List[int]]]:
    """

    This is the fundamental building block of the parallel combinatorial algorithm, here we branch off of a known feasible active set combination\\
    and then


    :param program: A multiparametric program
    :param active_set: the active set combination that we are expanding on
    :param murder_list: the list containing all previously found
    :param gen_children: A boolean flag, that determines if we should generate the children subsets
    :return: a list of the following form [Optional[CriticalRegion], pruned active set combination,Possibly Feasible Active set combinations]
    """
    t_set: Tuple[int, ...] = tuple(active_set)

    candidate_cr: Optional[CriticalRegion] = None
    pruned_active_sets: Set[Tuple[int,...]] = set()
    child_active_sets: List[List[int]] = []

    # return_list: Tuple[CriticalRegion, Set, List] = [None, set(), []]

    is_feasible_ = program.check_feasibility(active_set)

    if not is_feasible_:
        pruned_active_sets.add(t_set)
        return candidate_cr, pruned_active_sets, child_active_sets

    is_optimal_ = program.check_optimality(active_set)  # is_optimal(program, equality_indices)

    if not is_optimal_:
        if gen_children:
            child_active_sets = generate_children_sets(active_set, program.num_constraints(), murder_list)
        return candidate_cr, pruned_active_sets, child_active_sets

    candidate_cr = gen_cr_from_active_set(program, active_set)

    if candidate_cr is None:
        pruned_active_sets.add(t_set)
        return candidate_cr, pruned_active_sets, child_active_sets

    if gen_children:
        child_active_sets = generate_children_sets(active_set, program.num_constraints(), murder_list)

    return candidate_cr, pruned_active_sets, child_active_sets


def solve(program: MPQP_Program, num_cores=-1) -> Solution:
    """
    Solves the MPQP program with a modified algorithm described in Gupta et al. 2011

    This is the parallel version of the combinatorial.

    url: https://www.sciencedirect.com/science/article/pii/S0005109811003190

    :param num_cores: Sets the number of cores that are allocated to run this algorithm
    :param program: MPQP to be solved
    :return: the solution of the MPQP
    """
    # thread pool that we will be using
    start = time.time()

    if num_cores == -1:
        num_cores = num_cpu_cores()

    print(f'Spawned threads across {num_cores}')

    pool = Pool(num_cores)

    murder_list = CombinationTester()

    to_check = []

    solution = Solution(program, [])

    max_depth = max(program.num_x(), program.num_t()) - len(program.equality_indices)

    # breath first to increase efficiency of elimination
    root_node = generate_children_sets(program.equality_indices, program.num_constraints())

    to_check.extend(root_node)

    for i in range(max_depth):
        print(f'Time at depth test {i + 1}, {time.time() - start}')
        print(f'Number of active sets to be considered is {len(to_check)}')

        depth_time = time.time()

        gen_children = i + 1 != max_depth

        f = lambda x: full_process(program, x, murder_list, gen_children)

        future_list = []

        shuffle(to_check)

        outputs = pool.map(f, to_check)

        print(f'Time to run all tasks in parallel {time.time() - depth_time}')
        depth_time = time.time()

        if i + 1 == max_depth:
            for output in outputs:
                if output[0] is not None:
                    solution.add_region(output[0])
            break

        for output in outputs:
            murder_list.add_combos(output[1])
            future_list.extend(output[2])
            if output[0] is not None:
                solution.add_region(output[0])

        print(f'Time to process all depth outputs {time.time() - depth_time}')

        to_check = future_list

        # If there are not more active sets to check we are done
        if len(to_check) == 0:
            break

    # we never actually tested the program base active set
    if program.check_feasibility(program.equality_indices):
        if program.check_optimality(program.equality_indices):
            region = gen_cr_from_active_set(program, program.equality_indices)
            if region is not None:
                solution.add_region(region)

    pool.clear()

    return solution
