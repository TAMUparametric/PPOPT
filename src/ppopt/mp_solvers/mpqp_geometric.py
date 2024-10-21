
from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.general_utils import make_column
from ..utils.mpqp_utils import gen_cr_from_active_set
from .solver_utils import fathem_facet, get_facet_centers


def solve(program: MPQP_Program, active_set=None) -> Solution:
    """
    This solved the multiparametric program using the geometric algorithm described in Spjotvold et al.

    https://www.sciencedirect.com/science/article/pii/S1474667016369154

    :param program: a multiparametric program
    :param active_set: an initial optimal active set combination
    :return: the solution to the multiparametric optimization problem
    """
    if active_set is None:
        active_set = program.gen_optimal_active_set()
        print(f'Using a found active set {active_set}')

    initial_region = gen_cr_from_active_set(program, active_set, check_full_dim=False)

    if initial_region is None:
        return Solution(program, [])

    solution = Solution(program, [initial_region])
    solution_tol = solution.point_location_tolerance

    unchecked_regions = [initial_region]

    indexed_region_as = set()
    indexed_region_as.add(tuple(active_set))

    while len(unchecked_regions) > 0:

        # we want to expand from each region facet

        cur_region = unchecked_regions.pop(0)
        A, b = cur_region.E, cur_region.f

        # we want to look at every facet on the region
        facet_information = get_facet_centers(A, b, program.solver)

        for center, normal, radius in facet_information:

            # make sure we are pointing in the correct direction
            center_c = make_column(center)
            normal_c = make_column(normal)

            possible_cr = fathem_facet(center_c, normal_c, radius, program, indexed_region_as, cur_region.active_set, solution)

            if possible_cr is not None:
                indexed_region_as.add(tuple(possible_cr.active_set))
                unchecked_regions.append(possible_cr)
                solution.add_region(possible_cr)

    solution.point_location_tolerance = solution_tol

    return solution
