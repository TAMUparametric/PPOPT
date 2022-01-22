# this is the interface for the mpQP and mpLP problem solvers

from enum import Enum

from . import mpqp_geometric
from ..mp_solvers import mpqp_combinatorial
from ..mp_solvers import mpqp_graph
from ..mp_solvers import mpqp_parallel_geometric
from ..mp_solvers import mpqp_parallel_geometric_exp
from ..mp_solvers import mpqp_parrallel_combinatorial
from ..mp_solvers import mpqp_parrallel_combinatorial_exp
from ..mp_solvers import mpqp_parrallel_graph
from ..mpqp_program import MPQP_Program
from ..solution import Solution


class mpqp_algorithm(Enum):
    """
    Enum that selects the mpqp algorithm to be used

    This is done by passing the argument mpqp_algorithm.algorithm
    """
    combinatorial = 'combinatorial'
    combinatorial_parallel = 'p combinatorial'
    combinatorial_parallel_exp = 'p combinatorial exp'
    graph = 'graph'
    graph_exp = 'graph exp'
    graph_parallel = 'p graph'
    graph_parallel_exp = 'p graph exp'
    geometric = 'geometric'
    geometric_parallel = 'p geometric'
    geometric_parallel_exp = 'p geometric exp'


def solve_mpqp(problem: MPQP_Program, algorithm: mpqp_algorithm = mpqp_algorithm.combinatorial) -> Solution:
    """
    Takes a mpqp programming problem and solves it in a specified manner, In addition this solves MPLPs. The default
     solve algorithm is the Combinatorial algorithm by Gupta. et al.

    :param problem: Multiparametric Program to be solved
    :param algorithm: Selects the algorithm to be used
    :return: the solution of the MPQP, returns an empty solution if there is not an implemented algorithm
    """
    solution = Solution(problem, [])

    if algorithm is mpqp_algorithm.combinatorial:
        solution = mpqp_combinatorial.solve(problem)

    if algorithm is mpqp_algorithm.combinatorial_parallel:
        solution = mpqp_parrallel_combinatorial.solve(problem)

    if algorithm is mpqp_algorithm.combinatorial_parallel_exp:
        solution = mpqp_parrallel_combinatorial_exp.solve(problem)

    if algorithm is mpqp_algorithm.graph:
        solution = mpqp_graph.solve(problem)

    if algorithm is mpqp_algorithm.graph_exp:
        solution = mpqp_graph.solve_no_murder(problem)

    if algorithm is mpqp_algorithm.graph_parallel:
        solution = mpqp_parrallel_graph.solve(problem)

    if algorithm is mpqp_algorithm.graph_parallel_exp:
        solution = mpqp_parrallel_graph.solve_no_murder(problem)

    if algorithm is mpqp_algorithm.geometric:
        solution = mpqp_geometric.solve(problem)

    if algorithm is mpqp_algorithm.geometric_parallel:
        solution = mpqp_parallel_geometric.solve(problem)

    if algorithm is mpqp_algorithm.geometric_parallel_exp:
        solution = mpqp_parallel_geometric_exp.solve(problem)

    return filter_solution(solution)


def filter_solution(solution: Solution) -> Solution:
    """
    This is a place holder function, in the future this will be used to process and operate on the solution before it is returned to the user

    :param solution: a multi parametric solution

    :return: A processed solution
    """
    # filtered_solution = Solution(solution.program, [])
    #
    # for region in solution.critical_regions:
    #    if get_chebyshev_information(region).sol[-1] > 10 ** -5:
    #         filtered_solution.add_region(region)

    return solution
