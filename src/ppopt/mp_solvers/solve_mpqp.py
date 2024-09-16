# this is the interface for the mpQP and mpLP problem solvers

from enum import Enum

import numpy

from ..mp_solvers import (
    mpqp_combi_graph,
    mpqp_combinatorial,
    mpqp_graph,
    mpqp_parallel_combinatorial_exp,
    mpqp_parallel_geometric,
    mpqp_parallel_geometric_exp,
    mpqp_parrallel_combinatorial,
    mpqp_parrallel_graph,
)
from ..mplp_program import MPLP_Program
from ..mpqp_program import MPQP_Program
from ..solution import Solution
from . import mpqp_geometric


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
    combinatorial_graph = 'combinatorial graph'
    geometric = 'geometric'
    geometric_parallel = 'p geometric'
    geometric_parallel_exp = 'p geometric exp'

    def __str__(self):
        return self.name

    @staticmethod
    def all_algos():
        output = ''
        for algo in mpqp_algorithm:
            output += f'mpqp_algorithm.{algo}\n'
        return output


def solve_mpqp(problem: MPQP_Program, algorithm: mpqp_algorithm = mpqp_algorithm.combinatorial) -> Solution:
    """
    Takes a mpqp programming problem and solves it in a specified manner, In addition this solves MPLPs. The default
     solve algorithm is the Combinatorial algorithm by Gupta. et al.

    :param problem: Multiparametric Program to be solved
    :param algorithm: Selects the algorithm to be used
    :return: the solution of the MPQP, returns an empty solution if there is not an implemented algorithm
    """

    if not isinstance(algorithm, mpqp_algorithm):
        raise TypeError(
            f"You must pass an algorithm from mpqp_algorithm as the continuous algorithm. These can be found by "
            f"importing the following \n\nfrom ppopt.mp_solvers.solve_mpqp import mpqp_algorithm\n\nWith the "
            f"following choices\n{mpqp_algorithm.all_algos()}")

    solution = Solution(problem, [])

    if algorithm is mpqp_algorithm.combinatorial:
        solution = mpqp_combinatorial.solve(problem)

    if algorithm is mpqp_algorithm.combinatorial_parallel:
        solution = mpqp_parrallel_combinatorial.solve(problem)

    if algorithm is mpqp_algorithm.combinatorial_parallel_exp:
        solution = mpqp_parallel_combinatorial_exp.solve(problem)

    if algorithm is mpqp_algorithm.graph:
        solution = mpqp_graph.solve(problem)

    if algorithm is mpqp_algorithm.graph_exp:
        solution = mpqp_graph.solve(problem, use_pruning=False)

    if algorithm is mpqp_algorithm.graph_parallel:
        solution = mpqp_parrallel_graph.solve(problem)

    if algorithm is mpqp_algorithm.graph_parallel_exp:
        solution = mpqp_parrallel_graph.solve(problem, use_pruning=False)

    if algorithm is mpqp_algorithm.geometric:
        solution = mpqp_geometric.solve(problem)

    if algorithm is mpqp_algorithm.geometric_parallel:
        solution = mpqp_parallel_geometric.solve(problem)

    if algorithm is mpqp_algorithm.geometric_parallel_exp:
        solution = mpqp_parallel_geometric_exp.solve(problem)

    if algorithm is mpqp_algorithm.combinatorial_graph:
        solution = mpqp_combi_graph.solve(problem)

    # check if there needs to be a flag thrown in the case of overlapping critical regions
    # happens if there are negative or zero eigen values for mpQP (kkt conditions can find a lot of saddle points)
    if isinstance(problem, MPQP_Program):
        if min(numpy.linalg.eigvalsh(problem.Q)) <= 0:
            solution.is_overlapping = True

    # in the case of degenerate problems there are overlapping critical regions, unless a check is performed to prove
    # no overlap it is generally safer to consider that the mpLP case is overlapping
    if isinstance(problem, MPLP_Program):
        solution.is_overlapping = True

    return filter_solution(solution)


def filter_solution(solution: Solution) -> Solution:
    """
    This is a placeholder function, in the future this will be used to process and operate on the solution before it
    is returned to the user.

    :param solution: a multi parametric solution

    :return: A processed solution
    """

    return solution
