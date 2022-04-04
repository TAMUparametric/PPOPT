from typing import Optional, List

import numpy
# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool

from .solver_utils import get_facet_centers
from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.general_utils import make_column, num_cpu_cores
from ..utils.mpqp_utils import gen_cr_from_active_set


def fathem_facet_exp(center: numpy.ndarray, normal: numpy.ndarray, radius: float, program, current_active_set: list) -> \
        Optional[List]:
    # make sure we are pointing in the correct direction
    center = make_column(center)
    normal = make_column(normal)

    # make sure we are pointing in the correct dimension
    dist = radius * (10 ** (-6))

    while dist < radius:

        dist *= 2

        test_point = normal * dist + center

        sol = program.solve_theta(test_point)

        # test to see if the theta substituted optimization function is not feasible
        # this happens when we are looking outside the feasible space -> no longer need to look further
        if sol is None:
            return None

        # grab the active set
        # noinspection PyTypeChecker
        projected_set: List[int] = sol.active_set.tolist()

        # test for accidental self inclusion
        if projected_set == current_active_set:
            continue

        if not program.check_active_set_rank(projected_set):
            continue

        # indexed_region_as.add(tuple(projected_set))
        return projected_set

    # if no CR found return None
    return None


def full_process_2(program, current_active_set):
    if program.check_optimality(current_active_set) is None:
        return None

    critical_region = gen_cr_from_active_set(program, current_active_set, check_full_dim=True)

    if critical_region is None:
        return None

    return critical_region, get_facet_centers(critical_region.E, critical_region.f)


def fathem_initial_active_sets(program: MPQP_Program, initial_active_sets: List[List[int]] = None):
    """
    Covers an initial active set

    :param program:
    :param initial_active_sets:
    :return:
    """
    cr_gen = lambda active_set: gen_cr_from_active_set(program=program, active_set=active_set, check_full_dim=True)

    crs = [cr for cr in map(cr_gen, initial_active_sets) if cr is not None]

    work_items = []

    for cr in crs:
        facets = get_facet_centers(cr.E, cr.f)
        work_items.extend([(theta, facet_normal, radius, cr.active_set) for theta, facet_normal, radius in facets])

    return work_items, crs


def solve(program: MPQP_Program, initial_active_sets: List[List[int]] = None, num_cores=-1) -> Optional[Solution]:
    """
    This solved the multiparametric program using the geometric algorithm described in Spjotvold et al.

    https://www.sciencedirect.com/science/article/pii/S1474667016369154

    :param program: a multiparametric program
    :param initial_active_sets: a set of optimal active set combinations to initiate the algorithm
    :param num_cores: number of cores to run this calculation on, default of -1 means use all available cores
    :return: the solution to the multiparametric optimization problem
    """
    if initial_active_sets is None:
        initial_active_sets = [program.gen_optimal_active_set()]
        print(f'Using a found active set {initial_active_sets[-1]}')

    # initial_region = gen_cr_from_active_set(program, initial_active_sets[-1], check_full_dim=False)

    if num_cores == -1:
        num_cores = num_cpu_cores()

    print(f'Spawned threads across {num_cores}')

    pool = Pool(num_cores)

    cr_gen = lambda active_set: gen_cr_from_active_set(program=program, active_set=active_set, check_full_dim=True)

    facet_gen = lambda cr: get_facet_centers(cr.E, cr.f)

    initial_critical_regions = [cr for cr in pool.map(cr_gen, initial_active_sets) if cr is not None]
    initial_facets = pool.map(facet_gen, initial_critical_regions)

    work_items = []
    for facets, region in zip(initial_facets, initial_critical_regions):
        work_items.extend([(theta, facet_normal, radius, region.active_set) for theta, facet_normal, radius in facets])

    solution = Solution(program, initial_critical_regions)

    indexed_region_as = set()

    for region in initial_critical_regions:
        indexed_region_as.add(tuple(region.active_set))

    while len(work_items) > 0:

        print(f' Number of Facets to look at this time {len(work_items)}')
        f = lambda x: fathem_facet_exp(x[0], x[1], x[2], program, x[3])

        found_active_sets = pool.map(f, work_items)
        work_items = [active_set for active_set in found_active_sets if active_set is not None]
        work_items = [active_set for active_set in work_items if (tuple(active_set)) not in indexed_region_as]
        work_items = list(set([tuple(active_set) for active_set in work_items]))
        work_items = [list(active) for active in work_items]
        f = lambda x: full_process_2(program, x)

        outputs = pool.map(f, work_items)
        work_items = []
        # process the outputs
        filtered_outputs = [output for output in outputs if output is not None]
        print(f' Number of Regions adding in this pass {len(filtered_outputs)}!')

        for output in filtered_outputs:

            found_cr = output[0]
            facets = output[1]

            # check to see if we have found this region before
            if tuple(found_cr.active_set) not in indexed_region_as:
                # if we haven't added it to the active set index
                indexed_region_as.add(tuple(found_cr.active_set))
                # add it to the solution
                solution.add_region(found_cr)
                # add the associated work items from the facets to the queue
                work_items.extend(
                    [(theta, facet_normal, radius, found_cr.active_set) for theta, facet_normal, radius in facets])

    pool.clear()

    return solution
