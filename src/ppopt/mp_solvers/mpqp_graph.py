from typing import List

from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.constraint_utilities import is_full_rank
from ..utils.mpqp_utils import gen_cr_from_active_set
from .solver_utils import CombinationTester, generate_extra, generate_reduce


def graph_initialization(program, initial_active_sets):
    """
    Initializes the graph algorithm based on input

    :param program:
    :param initial_active_sets:
    :return:
    """
    if initial_active_sets is None:
        initial_active_sets = program.sample_theta_space()

    # This will contain all the attempted active sets
    attempted = set()

    solution = Solution(program, [])

    murder_list = CombinationTester()

    to_attempt = [tuple(a_set) for a_set in initial_active_sets]

    if len(to_attempt) != 0:
        print(f'First region {to_attempt[0]}')
    else:
        print('Failed to find an initial region!')

    return attempted, solution, murder_list, to_attempt


def solve(program: MPQP_Program, initial_active_sets: List[List[int]] = None, use_pruning: bool = True) -> Solution:
    """
    Solves the MPQP program with a modified algorithm described in Oberdieck et al. 2016

    url: https://www.sciencedirect.com/science/article/pii/S0005109816303971

    :param program: MPQP to be solved
    :param initial_active_sets: An initial critical region to start this algorithm with, otherwise one will be found
    :param use_pruning: Flag for setting if a pruning list is kept and utilized to remove
    :return: the solution of the MPQP
    """

    # TODO: This still misses some Critical Regions. USE Geometric Repair?

    attempted, solution, murder_list, to_attempt = graph_initialization(program, initial_active_sets)

    if not use_pruning:
        murder_list = None

    while len(to_attempt) > 0:

        # make sure I am grabbing from the lowest cardinality
        to_attempt.sort(key=len)
        # step 1: feasibility

        candidate = to_attempt.pop(0)
        # print(candidate)
        if candidate in attempted:
            continue

        attempted.add(candidate)

        # checks for infeasible subsets if so break and go to next candidate

        if not is_full_rank(program.A, list(candidate)):
            to_attempt.extend(generate_reduce(candidate, murder_list, attempted, set(program.equality_indices)))

            if murder_list is not None:
                murder_list.add_combo(candidate)
            continue

        if program.check_feasibility(list(candidate)) is None:
            to_attempt.extend(generate_reduce(candidate, murder_list, attempted, set(program.equality_indices)))

            if murder_list is not None:
                murder_list.add_combo(candidate)
            continue

        if not program.check_optimality(list(candidate)):
            to_attempt.extend(generate_reduce(candidate, murder_list, attempted, set(program.equality_indices)))
            # not optimal do nothing with this
            continue

        region = gen_cr_from_active_set(program, list(candidate), check_full_dim=False)

        if region is None:
            continue

        if region.is_full_dimension():
            solution.add_region(region)

            to_attempt.extend(generate_reduce(candidate, murder_list, attempted, set(program.equality_indices)))

            to_attempt.extend(generate_extra(candidate, region.regular_set[1], murder_list, attempted))

    return solution
#
#
# def solve_no_murder(program: MPQP_Program, initial_active_sets: List[List[int]] = None) -> Solution:
#     """
#     Solves the MPQP program with a modified algorithm described in Oberdieck et al. 2016
#
#     url: https://www.sciencedirect.com/science/article/pii/S0005109816303971
#
#
#     :param program: MPQP to be solved
#     :param initial_active_sets: An initial critical region to start this algorithm with, otherwise one will be found
#     :return: the solution of the MPQP
#     """
#
#     # TODO: This still misses some Critical Regions. USE Geometric Repair?
#
#     attempted, solution, _, to_attempt = graph_initialization(program, initial_active_sets)
#
#     while len(to_attempt) > 0:
#
#         # make sure I am grabbing from the lowest cardinality
#         to_attempt.sort(key=len)
#         # step 1: feasibility
#
#         candidate = to_attempt.pop(0)
#         # print(candidate)
#         if candidate in attempted:
#             continue
#
#         attempted.add(candidate)
#
#         # checks for infeasible subsets if so break and go to next candidate
#
#         if not is_full_rank(program.A, list(candidate)):
#             to_attempt.extend(generate_reduce(candidate, None, attempted, set(program.equality_indices)))
#             continue
#
#         if program.check_feasibility(list(candidate)) is None:
#             to_attempt.extend(generate_reduce(candidate, None, attempted, set(program.equality_indices)))
#             continue
#
#         if not program.check_optimality(list(candidate)):
#             to_attempt.extend(generate_reduce(candidate, None, attempted, set(program.equality_indices)))
#             # not optimal do nothing with this
#             continue
#
#         region = gen_cr_from_active_set(program, list(candidate), check_full_dim=False)
#
#         if region is None:
#             continue
#
#         if region.is_full_dimension():
#             solution.add_region(region)
#
#             to_attempt.extend(generate_reduce(candidate, None, attempted, set(program.equality_indices)))
#
#             to_attempt.extend(generate_extra(candidate, region.regular_set[1], None, attempted))
#
#     return solution
