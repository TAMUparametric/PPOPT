from typing import List, Optional, Set, Tuple, Union

import numpy

from ..critical_region import CriticalRegion
from ..mplp_program import MPLP_Program
from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..solver import Solver
from ..utils.chebyshev_ball import chebyshev_ball
from ..utils.general_utils import make_column
from ..utils.mpqp_utils import gen_cr_from_active_set


class CombinationTester:
    """Keeps track of all the infeasible active set combinations and filters prospective active set
    combinations """

    def __init__(self):
        """
        Initializes the combination tester with the following elements

        combos = empty set
        new_combos = empty set
        """
        self.combos = set()
        self.new_combos = set()

    def check(self, active_set: Set[int]) -> bool:
        """
        Checks if the provided active set combination is a superset of a previously tested infeasible active set

        :param active_set:
        :return: False if it should be culled and not tested any further, True if the set could be feasible
        """
        if not isinstance(active_set, set):
            active_set = set(active_set)

        if not active_set:
            return True

        for i in self.combos:
            if active_set.issuperset(i):
                return False

        return True

    def add_combo(self, active_set) -> None:
        if isinstance(active_set, tuple):
            self.combos.add(active_set)
        if not isinstance(active_set, set):
            self.combos.add(tuple(active_set))

    def add_combos(self, set_list: Set[Tuple[int]]) -> None:
        self.combos.update(set_list)


def manufacture_lambda(attempted, murder_list):
    if attempted is None:
        if murder_list is None:
            return lambda _: True
        else:
            return murder_list.check
    elif murder_list is None:
        return lambda x: x not in attempted
    else:
        return lambda x: x not in attempted and murder_list.check(x)


def generate_reduce(candidate: tuple, murder_list=None, attempted=None,
                    equality_set: Optional[Set[int]] = None) -> list:
    if equality_set is None:
        equality_set = set()

    check = manufacture_lambda(attempted, murder_list)

    accepted_sets = []

    for i in candidate:
        possible = tuple(sorted([j for j in candidate if j != i]))
        if check(possible) and set(possible).issuperset(equality_set):
            accepted_sets.append(possible)

    return accepted_sets


def generate_extra(candidate: tuple, expansion_set, murder_list=None, attempted=None) -> list:
    """
    Special routine for graph based algorithm, where we look for optimal active sets that

    :param candidate:
    :param expansion_set:
    :param murder_list:
    :param attempted:
    :return:
    """
    check = manufacture_lambda(attempted, murder_list)

    accepted_sets = []

    for regular_constraint in expansion_set:
        val = list(candidate)
        val.append(regular_constraint)
        future_child = tuple(sorted(val))
        if check(future_child):
            accepted_sets.append(future_child)

    return accepted_sets


def find_optimal_set(problem: Union[MPLP_Program, MPQP_Program]) -> List[int]:
    """
    This is a simple combinatorial algorithm for finding the first optimal active set. This is more or less only here for
    implementation completeness as this should never be used in practice.

    :param problem: a multiparametric optimization program
    :return: An optimal active set combination
    """
    optimal_set = []
    super_checker = CombinationTester()

    feasible_set = [problem.equality_indices]

    while True:

        to_check = []

        for active_set in feasible_set:

            optimality = problem.check_optimality(active_set)

            if optimality is not None:
                region = gen_cr_from_active_set(problem, active_set)
                if region is not None:
                    optimal_set = active_set

            else:
                to_check.extend(generate_children_sets(active_set, problem.num_constraints(), super_checker))

        feasible_set = to_check

        if len(feasible_set) == 0:
            break

        if len(feasible_set[0]) == max(problem.num_t(), problem.num_x()):
            break

        if len(optimal_set) != 0:
            break

    return optimal_set


def generate_children_sets(active_set, num_constraints: int, murder_list=None) -> List[List[int]]:
    # takes the active set and then generates all super sets of higher cardinality

    def check(x) -> bool:
        if murder_list is not None:
            return murder_list.check(x)
        else:
            return True

    if len(active_set) == 0:
        return [[i] for i in range(num_constraints) if check([i])]
    else:
        return [[*active_set, i] for i in range(active_set[-1] + 1, num_constraints) if check([*active_set, i])]


def find_sub_active_set(program: Union[MPLP_Program, MPQP_Program], active_set: List[int]) -> List[int]:
    """
    In the situation that there is an overdetermined active set we need to find the subset that is full rank and allows
    for generating active sets that are well defined.


    """

    eq_cons = program.equality_indices
    ineq_constraints = [i for i in active_set if i not in eq_cons]
    kept_inequalities = []

    current_rank = 0

    if len(eq_cons) == 0:
        current_rank = 0
    else:
        current_rank = numpy.linalg.matrix_rank(program.A[eq_cons])

    for i in ineq_constraints:

        trial_ineq = [*kept_inequalities, i]

        A_block = numpy.block([[program.A[eq_cons]], [program.A[trial_ineq]]])

        A_rank = numpy.linalg.matrix_rank(A_block)

        if A_rank > current_rank:
            kept_inequalities.append(i)
            current_rank = A_rank

        if current_rank == program.num_x():
            return [*eq_cons, *kept_inequalities]

    return [*eq_cons, *kept_inequalities]


def get_facet_centers(A: numpy.ndarray, b: numpy.ndarray, solver: Optional[Solver] = None) -> List[
    Tuple[numpy.ndarray, numpy.ndarray, float]]:
    r"""
    This takes the polytope P, and finds all the chebychev centers and normal vectors of each facet and the radius.

    .. math::
        P =  \{x \in \mathbb{R}^n: Ax \leq b\}

    :param A: The LHS constraint matrix
    :param b: The RHS constraint vector
    :return: a list with a tuple for each facet in the polytope (chebychev center, facet normal vector, chebychev radius)
    """
    facet_centers = []

    if solver is None:
        solver = Solver()

    for facet_index in range(A.shape[0]):

        # theta point to look out of
        theta = None
        radius = 0.0
        if A.shape[1] == 1:
            # if A is 1 dim then we can safely skip the chebyshev ball
            theta = numpy.array([[b[facet_index] / A[facet_index][0]]])
            radius = 1.0
        else:
            # We take the chebychev ball of the facet
            chev_ball = chebyshev_ball(A, b, [facet_index], deterministic_solver=solver.solvers['lp'])
            if chev_ball is not None:
                theta = chev_ball.sol[:-1]
                radius = chev_ball.sol[-1]
        if theta is None:
            continue

        # facet radius is numerically zero
        if abs(radius) <= 1e-12:
            continue

        facet_normal = make_column(A[facet_index])

        facet_centers.append((theta, facet_normal, radius))

    return facet_centers


def fathem_facet(center: numpy.ndarray, normal: numpy.ndarray, radius: float, program, indexed_region_as: Set,
                 current_active_set: list, cand_sol: Optional[Solution] = None) -> Optional[CriticalRegion]:
    """
    This explores the feasible space looking out from a chebychev center of a critical region facet.

    Starts from a point with a slight offset from the facet ;= center + D*normal and then check if this is feasible, check if for numerical reasons we accidentally hit ourselves,
    check to see if we stepped into a found region, if the active set is full rank we can try to build the critical region, return if full dimensional else double the distance from the facet we are considering

    :param center: chebychev center of
    :param normal: the normal of the polytope facet
    :param radius: chebychev radius of the polytope facet
    :param program: the multiparametric program being considered
    :param indexed_region_as: set of all indexed critical region active sets
    :param current_active_set: active set of the region we are stepping out of
    :param cand_sol: a candidate solution to check against
    :return: a critical region of the other side of the facet if one exists otherwise none
    """
    # ensure that our initial point, our normal are the vertical vectors and set up dist
    center = make_column(center)
    normal = make_column(normal)
    dist = radius * (10 ** (-6))

    # will run for at most 20 iterations before exiting
    while dist < radius:

        dist *= 2

        test_point = normal * dist + center

        # if we have a candidate solution
        if cand_sol is not None:
            # if our test point is inside a region we can safely break
            cand_sol.point_location_tolerance = 0.0
            if cand_sol.get_region(test_point) is not None:
                return None

        sol = program.solve_theta(test_point)

        # test to see if the theta substituted optimization function is not feasible
        # this happens when we are looking outside the feasible space -> no longer need to look further
        if sol is None:
            return None

        # grab the active set
        # noinspection PyTypeChecker
        projected_set: List[int] = sol.active_set.tolist()

        # if we have an overdetermined active set we need to find the subset that is full rank
        if len(projected_set) > program.num_x():
            projected_set = find_sub_active_set(program, projected_set)

        # test for accidental self inclusion
        if projected_set == current_active_set:
            continue

        # test if we are stepping into a found region
        if tuple(projected_set) in indexed_region_as:
            return None

        # we are not self intersecting, nor is this a know active set,
        # we have found a new critical region!
        # we need to check optimality (do we actually need to do this?)

        if not program.check_active_set_rank(projected_set):
            continue

        # build critical region
        cr = gen_cr_from_active_set(program, projected_set, check_full_dim=True)

        if cr is not None:
            return cr

    # if no CR found return None
    return None
