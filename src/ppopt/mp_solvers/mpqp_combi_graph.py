import numpy

from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.constraint_utilities import is_full_rank
from ..utils.mpqp_utils import gen_cr_from_active_set


def combinatorial_graph_initialization(program, initial_active_sets):
    """
    Initializes the graph algorithm based on input

    :param program:
    :param initial_active_sets:
    :return:
    """
    if initial_active_sets is None:
        initial_active_sets = program.sample_theta_space(1)

    # This will contain all the attempted active sets
    attempted = set()

    solution = Solution(program, [])

    to_attempt = set(sorted_tuple(a_set) for a_set in initial_active_sets)

    return attempted, solution, to_attempt


def sorted_tuple(x) -> tuple:
    """Helper function to make sure that inclusion tests are O(1)"""
    return tuple(sorted(x))


def remove_i(x, i: int) -> tuple:
    """helper function to remove an element i from a sorted tuple"""
    return sorted_tuple(set(x) - {i})


def add_i(x, i: int) -> tuple:
    """helper function to add an element i to a sorted tuple"""
    temp_x = set(x)
    temp_x.add(i)
    return sorted_tuple(temp_x)


def feasability_check(program: MPQP_Program, A) -> bool:
    A_x, b_x, A_l, b_l = program.optimal_control_law(list(A))

    # makes the assumption that the equality indices are at the top of the active set
    # as these can be any sign, we only care to enforce l(theta) >= 0
    A_l = A_l[len(program.equality_indices):]
    b_l = b_l[len(program.equality_indices):]

    N = program.num_constraints()
    A_ = program.A[[i for i in range(N) if i not in A]]
    b_ = program.b[[i for i in range(N) if i not in A]]
    F_ = program.F[[i for i in range(N) if i not in A]]

    # top constraints
    A_prob = numpy.block([[A_ @ A_x - F_], [-A_l], [program.A_t]])
    b_prob = numpy.block([[b_ - A_ @ b_x], [b_l], [program.b_t]])

    # build and solve the feasibility LP, if empty it will result in None
    return program.solver.solve_lp(None, A_prob, b_prob) is not None


def solve(program: MPQP_Program) -> Solution:
    """
    Solves the MPQP program with the joint combinatorial based connected graph approach of Arnström et al.

    This method removes the vast majority of the geometry calculations, leaving only the non-empty check of the critical region.

    url: https://arxiv.org/abs/2404.05511

    :param program: MPQP to be solved
    :return: the solution of the MPQP
    """
    # initialize with and Exclusion set, a base solution, and a set of things to visit
    E, solution, S = combinatorial_graph_initialization(program, None)

    # make sure that everything we have added to S is in E
    for s in S:
        E.add(sorted_tuple(s))

    def explore_subset(A_):
        """
        Explores the subset of the active set A_, does not allow removal of equality indicies
        :param A_: The active set that we are taking subsets of
        """
        for i in A_:
            if i not in program.equality_indices:
                A_trial = remove_i(A_, i)
                if A_trial not in E:
                    S.add(A_trial)
                    E.add(A_trial)

    def explore_superset(A_):
        """
        Explores the superset of the active set A_
        :param A_: The active set that we are taking super sets of
        """
        for i in range(program.num_constraints()):
            if i not in A_:
                A_trial = add_i(A_, i)
                if A_trial not in E:
                    S.add(A_trial)
                    E.add(A_trial)

    # while we have things to explore, we explore
    while len(S) > 0:


        # get an active set combination
        A = S.pop()

        print(A)

        # if we fail LINQ then we need to remove a constraint to hope to be full rank
        if not is_full_rank(program.A, list(A)):
            explore_subset(A)
        # if we are full rank, check if the resulting critical region is not empty
        elif feasability_check(program, A):

            # Attempts to construct the region
            cand_region = gen_cr_from_active_set(program, list(A))

            # if the candidate region is full dimensional
            if cand_region is not None:
                solution.add_region(cand_region)

            # adds the super, and subsets
            explore_subset(A)
            explore_superset(A)

    return solution
