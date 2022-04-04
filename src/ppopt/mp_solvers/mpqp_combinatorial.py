from typing import List

from .solver_utils import generate_children_sets, CombinationTester
from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.mpqp_utils import gen_cr_from_active_set


def solve(program: MPQP_Program) -> Solution:
    """
    Solves the MPQP program with a modified algorithm described in Gupta et al. 2011.
    The algorithm is described in this paper https://www.sciencedirect.com/science/article/pii/S0005109811003190

    :param program: MPQP to be solved
    :return: the solution of the MPQP
    """
    murder_list = CombinationTester()

    to_check = list()

    solution = Solution(program, [])

    max_depth = max(program.num_x(), program.num_t()) - len(program.equality_indices)
    # breath first to optimize the elimination

    root_node = generate_children_sets(program.equality_indices, program.num_constraints(), murder_list)

    to_check.extend(root_node)

    for i in range(max_depth):
        # if there are no other active sets to check break out of loop
        # print(len(to_check))

        future_sets = list()
        # creates the list of feasible active sets
        feasible_sets = check_child_feasibility(program, to_check, murder_list)

        for child_set in feasible_sets:

            # soln = check_optimality(program, equality_indices=child_set)
            # The active set is optimal try to build a critical region

            # if soln is not None:
            if program.check_optimality(child_set):
                critical_region = gen_cr_from_active_set(program, child_set)
                # Check the dimensions of the critical region
                if critical_region is not None:
                    solution.add_region(critical_region)

            # propagate sets

            if i + 1 != max_depth:
                future_sets.extend(generate_children_sets(child_set, program.num_constraints(), murder_list))

        to_check = future_sets

    if program.check_feasibility(program.equality_indices):
        if program.check_optimality(program.equality_indices):
            region = gen_cr_from_active_set(program, program.equality_indices)
            if region is not None:
                if region.is_full_dimension():
                    solution.add_region(region)

    return solution


def check_child_feasibility(program: MPQP_Program, set_list: List[List[int]], combination_checker: CombinationTester) -> \
        List[List[int]]:
    """
    Checks the feasibility of a list of active set combinations, if infeasible add to the combination checker and returns all feasible active set combinations

    :param program: An MPQP Program
    :param set_list: The list of active sets
    :param combination_checker: The combination checker that prunes
    :return: The list of all feasible active sets
    """
    output = list()
    for child in set_list:
        if program.check_feasibility(child):
            output.append(child)
        else:
            combination_checker.add_combo(child)

    return output
