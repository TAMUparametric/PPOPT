from itertools import permutations
from typing import List, Tuple, Optional
from collections import deque

import numpy
import copy

from ..mpmilp_program import MPMILP_Program
from ..critical_region import CriticalRegion

from .mpqp_utils import get_bounds_1d


def reduce_overlapping_critical_regions_1d(program: MPMILP_Program, regions: List[CriticalRegion]) -> Tuple[
    List[CriticalRegion], bool]:
    """
    This adjusts critical regions for the 1d-parameter case, so that there are no overlaps.

    :param program: the MPMILP_Program to be solved
    :param regions: a list of CriticalRegion objects
    :return: a list of CriticalRegion objects and a flag indicating whether there are still overlaps
    """

    # guard around the statement
    if program.num_t() != 1:
        raise ValueError('reduce_overlapping_critical_regions_1d requires a 1d-parameter problem')

    overlaps_remaining, regions = identify_overlaps(program, regions)

    return regions, overlaps_remaining


def identify_overlaps(program: MPMILP_Program, regions: List[CriticalRegion]) -> Tuple[bool, list]:
    new_regions = []
    to_remove = []
    possible_dual_degeneracy = False
    # test all permutations rather than combinations: this way we only need to test for 2 cases (CR 2 fully inside CR 1, or CR 1 left of CR 2 but overlapping), since the other 2 cases are handled by the swapped permutation
    to_check = deque(permutations(regions, 2))
    while len(to_check) > 0:

        region_added = False

        cr1, cr2 = to_check.popleft()
        # check if region has already been marked for removal, if so, no need to check again
        if cr1 in to_remove or cr2 in to_remove:
            continue

        # check if region 2 is fully inside region 1
        if full_overlap(cr1, cr2):
            if equal_linear_objective(program, outer_region=cr1, inner_region=cr2):
                # equal objective value over entire overlapping region indicates possible dual degeneracy
                # in this case, we keep both regions to have both solutions
                possible_dual_degeneracy = True
            elif region_dominates(program, cr1, cr2):
                to_remove.append(cr2)
            elif region_dominates(program, cr2, cr1):
                # objective of 2 always less than objective of 1 --> cr1_l, cr2, cr1_r
                new_regions, cr1 = split_outer_region(new_regions, outer_region=cr1, inner_region=cr2)
                region_added = True
            else:
                new_regions, cr2, cr1 = adjust_fully_overlapping_regions(program, new_regions, inner_region=cr2, outer_region=cr1)
                region_added = True
        # check if region 1 is to the left of region 2 but overlapping
        elif partial_overlap(cr1, cr2):
            # determine lower objective value in overlap and adjust region
            if region_dominates(program, cr1, cr2):
                ub1 = get_bounds_1d(cr1.E, cr1.f)[1]
                cr2 = cr_new_bounds(cr2, lb_new=ub1, ub_new=None)
            elif region_dominates(program, cr2, cr1):
                lb2 = get_bounds_1d(cr2.E, cr2.f)[0]
                cr1 = cr_new_bounds(cr1, lb_new=None, ub_new=lb2)
            else:
                intersection = compute_objective_intersection_point(program, cr1, cr2)[0]
                cr1 = cr_new_bounds(cr1, lb_new=None, ub_new=intersection)
                cr2 = cr_new_bounds(cr2, lb_new=intersection, ub_new=None)
        # if we added a new region, add all permutations with this region to the queue except for those with regions just checked (new region is a subset of those regions, no need to check again)
        if region_added:
            other_regions = [cr for cr in regions if cr not in [cr1, cr2] and cr not in to_remove]
            to_check.extend([r, new_regions[-1]] for r in other_regions)
            to_check.extend([new_regions[-1], r] for r in other_regions)

    # add new
    regions = [*regions, *new_regions]
    # purge
    regions = [cr for cr in regions if cr not in to_remove]

    return possible_dual_degeneracy, regions

def adjust_fully_overlapping_regions(program: MPMILP_Program, new_regions: List[CriticalRegion], inner_region: CriticalRegion, outer_region: CriticalRegion) -> Tuple[List[CriticalRegion], CriticalRegion, CriticalRegion]:
    intersection, expand_outer_on_left = compute_objective_intersection_point(program, outer_region, inner_region)
    new_regions.append(copy.deepcopy(outer_region))
    inner_lb, inner_ub = get_bounds_1d(inner_region.E, inner_region.f)
    if expand_outer_on_left:
        outer_region = cr_new_bounds(outer_region, lb_new=None, ub_new=intersection)
        inner_region = cr_new_bounds(inner_region, lb_new=intersection, ub_new=None)
        new_regions[-1] = cr_new_bounds(new_regions[-1], lb_new=inner_ub, ub_new=None)
    else:
        outer_region = cr_new_bounds(outer_region, lb_new=None, ub_new=inner_lb)
        inner_region = cr_new_bounds(inner_region, lb_new=None, ub_new=intersection)
        new_regions[-1] = cr_new_bounds(new_regions[-1], lb_new=intersection, ub_new=None)
    return new_regions, inner_region, outer_region

def split_outer_region(new_regions: List[CriticalRegion], outer_region: CriticalRegion, inner_region: CriticalRegion) -> Tuple[List[CriticalRegion], CriticalRegion]:
    inner_lb, inner_ub = get_bounds_1d(inner_region.E, inner_region.f)
    new_regions.append(copy.deepcopy(outer_region))
    outer_region = cr_new_bounds(outer_region, lb_new=None, ub_new=inner_lb)
    new_regions[-1] = cr_new_bounds(new_regions[-1], lb_new=inner_ub, ub_new=None)
    return new_regions, outer_region


def cr_new_bounds(cr: CriticalRegion, lb_new: Optional[float], ub_new: Optional[float]) -> CriticalRegion:
    lb, ub = get_bounds_1d(cr.E, cr.f)

    lb = lb if lb_new is None else lb_new
    ub = ub if ub_new is None else ub_new

    cr.E = numpy.array([[1], [-1]])
    cr.f = numpy.array([[ub], [-lb]])

    return cr


def full_overlap(cr1: CriticalRegion, cr2: CriticalRegion) -> bool:
    # region 2 fully inside region 1
    lb1, ub1 = get_bounds_1d(cr1.E, cr1.f)
    lb2, ub2 = get_bounds_1d(cr2.E, cr2.f)
    return lb1 <= lb2 and ub1 >= ub2


def partial_overlap(cr1: CriticalRegion, cr2: CriticalRegion) -> bool:
    # region 1 to the left of region 2
    lb1, ub1 = get_bounds_1d(cr1.E, cr1.f)
    lb2, ub2 = get_bounds_1d(cr2.E, cr2.f)
    return lb1 < lb2 and ub1 > lb2 and ub2 > ub1


def equal_linear_objective(program: MPMILP_Program, inner_region: CriticalRegion, outer_region: CriticalRegion) -> bool:
    f_inner_lb, f_inner_ub, f_outer_lb, f_outer_ub = evaluate_objective_at_middle_bounds(program, inner_region,
                                                                                         outer_region)
    return f_inner_lb == f_outer_lb and f_inner_ub == f_outer_ub

def region_dominates(program: MPMILP_Program, cr_1: CriticalRegion, cr_2: CriticalRegion) -> bool:
    """
    Check if cr_1 dominates cr_2 in the sense that the objective value of cr_1 is always less than or equal to that of cr_2.
    """
    f_1_lower, f_1_upper, f_2_lower, f_2_upper = evaluate_objective_at_middle_bounds(program, cr_1, cr_2)
    return f_1_lower <= f_2_lower and f_1_upper <= f_2_upper

# cr_1 is the outer region or the left region, depending on calling context
def compute_objective_intersection_point(program: MPMILP_Program, cr_1: CriticalRegion, cr_2: CriticalRegion) -> Tuple[
    float, bool]:
    f_1_lower, f_1_upper, f_2_lower, f_2_upper = evaluate_objective_at_middle_bounds(program, cr_1, cr_2)
    deltaf_1 = f_1_upper - f_1_lower
    deltaf_outer = f_2_upper - f_2_lower
    lower, upper = find_middle_bounds(cr_1, cr_2)
    delta = upper - lower
    return (f_2_lower - f_1_lower) / (deltaf_1 - deltaf_outer) * delta + lower, f_1_lower < f_2_lower


def evaluate_objective_at_middle_bounds(program: MPMILP_Program, cr_1: CriticalRegion, cr_2: CriticalRegion) -> Tuple[
    float, float, float, float]:
    lower, upper = find_middle_bounds(cr_1, cr_2)
    f_1_lower = program.evaluate_objective(cr_1.evaluate(numpy.array([[lower]])), numpy.array([[lower]]))
    f_1_upper = program.evaluate_objective(cr_1.evaluate(numpy.array([[upper]])), numpy.array([[upper]]))
    f_2_lower = program.evaluate_objective(cr_2.evaluate(numpy.array([[lower]])), numpy.array([[lower]]))
    f_2_upper = program.evaluate_objective(cr_2.evaluate(numpy.array([[upper]])), numpy.array([[upper]]))
    return f_1_lower, f_1_upper, f_2_lower, f_2_upper


def find_middle_bounds(cr_1: CriticalRegion, cr_2: CriticalRegion) -> Tuple[float, float]:
    lb_1, ub_1 = get_bounds_1d(cr_1.E, cr_1.f)
    lb_2, ub_2 = get_bounds_1d(cr_2.E, cr_2.f)
    values = [lb_1, ub_1, lb_2, ub_2]
    values.sort()
    return values[1], values[2]
