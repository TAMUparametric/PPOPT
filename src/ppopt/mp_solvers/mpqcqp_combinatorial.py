from typing import List

from ..mpqcqp_program import MPQCQP_Program
from ..mplp_program import MPLP_Program
from ..solution import Solution
# from ..utils.mpqp_utils import gen_cr_from_active_set
from .solver_utils import CombinationTester, generate_children_sets


def solve(program: MPQCQP_Program) -> Solution:
    """
    Solves the MPQCQP program with a modified algorithm described in Gupta et al. 2011.
    The algorithm is described in this paper https://www.sciencedirect.com/science/article/pii/S0005109811003190

    :param program: MQCPQP to be solved
    :return: the solution of the MPQCQP
    """
    murder_list = CombinationTester()

    to_check = []

    solution = Solution(program, [])

    max_depth = program.num_x() - len(program.equality_indices)
    # breath first to optimize the elimination

    root_node = generate_children_sets(program.equality_indices, program.num_constraints(), murder_list)

    to_check.extend(root_node)

    for i in range(max_depth):
        # if there are no other active sets to check break out of loop
        # print(len(to_check))

        future_sets = []
        # creates the list of feasible active sets

        # if this is a mpLP we can do a reduction on the active sets
        # this is not required but will give a perf boost
        if type(program) is MPLP_Program:
            condition = lambda child: child[-1] >= len(child) + program.num_constraints() - program.num_x()
            to_check = [child for child in to_check if not condition(child)]

        feasible_sets = check_child_feasibility(program, to_check, murder_list)

        for child_set in feasible_sets:

            # soln = check_optimality(program, equality_indices=child_set)
            # The active set is optimal try to build a critical region

            # if soln is not None:
            if program.check_optimality(child_set):
                critical_region_list = program.gen_cr_from_active_set(child_set)
                # Check the dimensions of the critical region
                if critical_region_list is not None:
                    for critical_region in critical_region_list:
                        solution.add_region(critical_region)

            # propagate sets

            if i + 1 != max_depth:
                future_sets.extend(generate_children_sets(child_set, program.num_constraints(), murder_list))

        to_check = future_sets

    if program.check_feasibility(program.equality_indices):
        if program.check_optimality(program.equality_indices):
            region_list = program.gen_cr_from_active_set(program.equality_indices)
            if region_list is not None:
                for region in region_list:
                    if region.is_full_dimension():
                        solution.add_region(region)

    return solution


def check_child_feasibility(program: MPQCQP_Program, set_list: List[List[int]], combination_checker: CombinationTester) -> \
        List[List[int]]:
    """
    Checks the feasibility of a list of active set combinations, if infeasible add to the combination checker and returns all feasible active set combinations

    :param program: An MPQP Program
    :param set_list: The list of active sets
    :param combination_checker: The combination checker that prunes
    :return: The list of all feasible active sets
    """
    output = []
    for child in set_list:
        if program.check_feasibility(child):
            output.append(child)
        else:
            combination_checker.add_combo(child)

    return output
