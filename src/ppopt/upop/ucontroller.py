# The purpose of this is to generate C++ code for to use for microcontroller but
# There are hopes to expand to further outreach such as generating dlls or .so
# or other language output to use solutions in other languages without needing a
# new ppopt in every language

from typing import List

import numpy

from ..critical_region import CriticalRegion
from ..solution import Solution
from ..solver_interface.solver_interface import solve_lp
from ..utils.general_utils import make_column


def determine_hyperplane(regions: List[CriticalRegion], hyper_planes: numpy.ndarray):
    """
    Finds the 'best' splitting hyper plane for this task.

    In this case best means minimizing the number of intersected regions while also maximizing the difference between
    supported and not supported regions.

    :param regions:
    :param hyper_planes:
    :return: []
    """
    best_index = 0
    best_support = list()
    best_not_support = list()
    best_intersection = list()

    best_diff = 10 * len(regions)
    best_over = 10 * len(regions)

    # TODO: Implement redundant hyperplane removal for speedup

    # remove_hyper_plan = list()

    for i in range(hyper_planes.shape[0]):

        support = list()
        not_support = list()
        intersection = list()

        for j, region in enumerate(regions):
            type_ = classify_polytope(region, hyper_planes[i])

            if type_ == 1:
                support.append(j)
            elif type_ == 0:
                intersection.append(j)
            elif type_ == -1:
                not_support.append(j)

        diff = abs(len(support) - len(not_support))
        over = len(intersection)

        if diff + over < best_diff + best_over and len(support) > 0 and len(not_support) > 0:
            best_index = i
            best_diff = diff
            best_over = over
            best_support = support
            best_not_support = not_support
            best_intersection = intersection

    return [best_index, best_support, best_not_support, best_intersection]


def classify_polytope(region: CriticalRegion, hyper_plane: numpy.ndarray) -> int:
    """
    We are going to classify the polytopic critical region by solving 2 LPS \n

    max ||<x,A>||-d for x in Critical region \n
    min ||<x,A>||-d for x in Critical region \n

    The result of the objective function will tell us the side of the hyper plane the point is on.

    :param region: Critical region
    :param hyper_plane: A fundamental hyperplane
    :return: -1 if completely not in support, 0 if intersected, 1 if completely in support
    """

    # form the needed matrices
    c = make_column(hyper_plane[0:region.E.shape[1]])
    d = hyper_plane[-1]

    # solve the minimization LP
    min_sol = solve_lp(c, region.E, region.f)
    max_sol = solve_lp(-c, region.E, region.f)

    # extract the objective values

    min_obj = min_sol.obj - d
    max_obj = max_sol.obj - d

    # of both are in support return 1 for in support
    if min_obj > 0 and max_obj > 0:
        return 1
    # if neither is in support return -1 for not in support
    elif min_obj < 0 and min_obj < 0:
        return -1
    # it is intersecting this region return 0 for intersection
    else:
        return 0


class BVH:
    """
    This is the Bounding Volume Hierarchy (BVH) class that decomposes the space that allows point location acceleration
    """

    def __init__(self, parent, fundamental_list, region_list, depth, index):
        """Initializes the BVH based on a recursive constructor."""
        self.depth = depth
        self.is_leaf = False
        self.region = -1
        self.parent = parent
        self.right_pos = -1
        self.left_pos = -1

        if len(region_list) == 1:
            self.region = region_list[0]
            self.is_leaf = True
            self.count = 1
        else:
            pass


def generate_code(solution: Solution) -> List[str]:
    """
    Generates C++17 code for point location and function evaluation on microcontrollers. This forms a BVH to
    accelerate solution times. WARNING: This breaks down at high dimensions.

    :param solution: a solution to a MPLP or MPQP solution
    :return: List of the strings of the C++17 datafiles that integrate with uPOP
    """

    # TODO: Finish Implementation

    # fundamental_c, original_c, parity_c = find_unique_region_hyperplanes(solution)

    # fundamental_f, original_f, parity_f = find_unique_region_functions(solution)

    return ["None"]
