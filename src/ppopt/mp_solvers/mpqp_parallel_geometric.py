from typing import Optional

import numpy
# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool

from .solver_utils import get_facet_centers, fathem_facet
from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.general_utils import num_cpu_cores
from ..utils.mpqp_utils import gen_cr_from_active_set


def full_process(center: numpy.ndarray, norm: numpy.ndarray, radius: float, program: MPQP_Program, current_active_set,
                 indexed_region_as):
    """
    This is the function block to be executed in parallel. Takes in a facet. Returns the associated CR on the other side of the facet
    if it exists, and all the facets associated with the other side of the

    :param center: Chebychev Center of a Critical Region Facet
    :param norm: Normal of the Facet
    :param radius: Chebychev Radius of the Critical Region Facet
    :param program: the multiparametric program being considered
    :param indexed_region_as: set of all critical regions found so far
    :param current_active_set: list of the active set that we are stepping out of
    :return: The identified Critical Region on the other side of the facet, and the facets of this critical region or None of nothing
    """
    found_solution = fathem_facet(center, norm, radius, program, current_active_set, indexed_region_as)

    # this will return a None
    if found_solution is None:
        return None

    return found_solution, get_facet_centers(found_solution.E, found_solution.f)


def solve(program: MPQP_Program, active_set=None, num_cores=-1) -> Optional[Solution]:
    """
    This solved the multiparametric program using the geometric algorithm described in Spjotvold et al.

    https://www.sciencedirect.com/science/article/pii/S1474667016369154

    :param program: a multiparametric program
    :param active_set: an initial optimal active set combination
    :param num_cores: number of cores to run this calculation on, default of -1 means use all available cores
    :return: the solution to the multiparametric optimization problem
    """
    if active_set is None:
        active_set = program.gen_optimal_active_set()
        print(f'Using a found active set {active_set}')

    initial_region = gen_cr_from_active_set(program, active_set, check_full_dim=False)

    if num_cores == -1:
        num_cores = num_cpu_cores()

    print(f'Spawned threads across {num_cores}')

    pool = Pool(num_cores)

    solution = Solution(program, [initial_region])

    indexed_region_as = set()
    indexed_region_as.add(tuple(list(active_set)))

    # initiate by exploring first region

    work_items = [(theta, facet_normal, radius, initial_region.active_set) for theta, facet_normal, radius in
                  get_facet_centers(initial_region.E, initial_region.f)]

    while len(work_items) > 0:

        print(f' Number of Facets to look at this time {len(work_items)}')
        f = lambda x: full_process(x[0], x[1], x[2], program, x[3], indexed_region_as)

        outputs = pool.map(f, work_items)

        # clear out the work queue
        work_items = []

        # process the outputs
        for output in outputs:

            if output is None:
                continue

            found_cr = output[0]
            facets = output[1]

            # check to see if we need to do anything
            if found_cr is not None:
                # we have a critical region!

                # check to see if we have found this region before
                if tuple(found_cr.active_set) not in indexed_region_as:
                    # if we haven't added it to the active set index
                    indexed_region_as.add(tuple(found_cr.active_set))
                    # add it to the solution
                    solution.add_region(found_cr)
                    # add the associated work items from the facets to the queue
                    work_items.extend(
                        [(theta, facet_normal, radius, found_cr.active_set) for theta, facet_normal, radius in facets])

    return solution
