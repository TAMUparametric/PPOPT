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

    overlaps_remaining, regions = identify_overlaps_1d(program, regions)

    return regions, overlaps_remaining


def identify_overlaps_1d(program: MPMILP_Program, regions: List[CriticalRegion]) -> Tuple[bool, list]:
    """
    Identifies overlaps between critical regions for the 1d-parameter case and adjusts the regions where possible.
    If dual degeneracy is detected, the corresponding overlap is not reduced and so, it is possible for the fianl list of regions to still contain overlaps.

    :param program: the MPMILP_Program to be solved
    :param regions: a list of CriticalRegion objects
    :return: a boolean indicating whether there are still overlaps and the updated list of CriticalRegion objects
    """
    new_regions = []
    to_remove = []
    possible_dual_degeneracy = False
    # test all permutations rather than combinations: this way we only need to test for 2 cases (CR 2 fully inside CR 1, or CR 1 left of CR 2 but overlapping), since the other 2 cases are handled by the swapped permutation
    to_check = deque(permutations(regions, 2))
    while len(to_check) > 0:

        region_added = False

        cr_1, cr_2 = to_check.popleft()
        # check if region has already been marked for removal, if so, no need to check again
        if cr_1 in to_remove or cr_2 in to_remove:
            continue

        # check if region 2 is fully inside region 1
        if full_overlap(cr_1, cr_2):
            if equal_linear_objective(program, cr_1, cr_2):
                # equal objective value over entire overlapping region indicates possible dual degeneracy
                # in this case, we keep both regions to have both solutions
                possible_dual_degeneracy = True
            elif region_dominates(program, cr_1, cr_2):
                to_remove.append(cr_2)
            elif region_dominates(program, cr_2, cr_1):
                # objective of 2 always less than objective of 1 --> cr_1_l, cr_2, cr_1_r
                new_regions, cr_1 = split_outer_region(new_regions, outer_region=cr_1, inner_region=cr_2)
                region_added = True
            else:
                new_regions, cr_2, cr_1 = adjust_fully_overlapping_regions(program, new_regions, inner_region=cr_2, outer_region=cr_1)
                region_added = True
        # check if region 1 is to the left of region 2 but overlapping
        elif partial_overlap(cr_1, cr_2):
            # determine lower objective value in overlap and adjust region
            if region_dominates(program, cr_1, cr_2):
                ub1 = get_bounds_1d(cr_1.E, cr_1.f)[1]
                cr_2 = cr_new_bounds(cr_2, lb_new=ub1, ub_new=None)
            elif region_dominates(program, cr_2, cr_1):
                lb2 = get_bounds_1d(cr_2.E, cr_2.f)[0]
                cr_1 = cr_new_bounds(cr_1, lb_new=None, ub_new=lb2)
            else:
                intersection = compute_objective_intersection_point(program, cr_1, cr_2)[0]
                cr_1 = cr_new_bounds(cr_1, lb_new=None, ub_new=intersection)
                cr_2 = cr_new_bounds(cr_2, lb_new=intersection, ub_new=None)
        # if we added a new region, add all permutations with this region to the queue except for those with regions just checked (new region is a subset of those regions, no need to check again)
        if region_added:
            other_regions = [cr for cr in regions if cr not in [cr_1, cr_2] and cr not in to_remove]
            to_check.extend([r, new_regions[-1]] for r in other_regions)
            to_check.extend([new_regions[-1], r] for r in other_regions)

    # add new
    regions = [*regions, *new_regions]
    # purge
    regions = [cr for cr in regions if cr not in to_remove]

    # remove numerically nil regions
    region_bounds = [get_bounds_1d(cr.E, cr.f) for cr in regions]
    regions = [cr for cr, bounds in zip(regions, region_bounds) if abs(bounds[0]- bounds[1]) > 1e-8]

    return possible_dual_degeneracy, regions

def adjust_fully_overlapping_regions(program: MPMILP_Program, new_regions: List[CriticalRegion], inner_region: CriticalRegion, outer_region: CriticalRegion) -> Tuple[List[CriticalRegion], CriticalRegion, CriticalRegion]:
    """
    Adjusts the bounds of two fully overlapping critical regions, when their objective functions interesect at some point.
    This also creates a new critical region, so that the outer region is split into two disjunct regions, with the updated inner region in between.

    :param program: the MPMILP_Program to be solved
    :param new_regions: a list of CriticalRegion objects, to which the newly created region will be added
    :param inner_region: the inner region
    :param outer_region: the outer region
    :return: the updated list of new critical regions to which one new region has been added, and the updated inner and outer regions
    """
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
    """
    Splits the outer region into two disjunct regions, with the inner region in between. Thereby, a new critical region is created.

    :param new_regions: a list of CriticalRegion objects, to which the newly created region will be added
    :param outer_region: the outer region
    :param inner_region: the inner region
    :return: the updated list of new critical regions to which one new region has been added, and the updated outer region
    """
    inner_lb, inner_ub = get_bounds_1d(inner_region.E, inner_region.f)
    new_regions.append(copy.deepcopy(outer_region))
    outer_region = cr_new_bounds(outer_region, lb_new=None, ub_new=inner_lb)
    new_regions[-1] = cr_new_bounds(new_regions[-1], lb_new=inner_ub, ub_new=None)
    return new_regions, outer_region


def cr_new_bounds(cr: CriticalRegion, lb_new: Optional[float], ub_new: Optional[float]) -> CriticalRegion:
    """
    Updates the bounds of a critical region.

    :param cr: the critical region
    :param lb_new: the new lower bound; if None, the lower bound is not updated
    :param ub_new: the new upper bound; if None, the upper bound is not updated
    :return: the CriticalRegion object with the updated bounds
    """
    lb, ub = get_bounds_1d(cr.E, cr.f)

    lb = lb if lb_new is None else lb_new
    ub = ub if ub_new is None else ub_new

    cr.E = numpy.array([[1], [-1]])
    cr.f = numpy.array([[ub], [-lb]])

    return cr


def full_overlap(cr_1: CriticalRegion, cr_2: CriticalRegion) -> bool:
    """
    Checks if the second critical region is fully inside the first critical region.

    :param cr_1: the first critical region
    :param cr_2: the second critical region
    :return: a boolean indicating whether the second critical region is fully inside the first critical region
    """

    # region 2 fully inside region 1
    lb1, ub1 = get_bounds_1d(cr_1.E, cr_1.f)
    lb2, ub2 = get_bounds_1d(cr_2.E, cr_2.f)
    return lb1 <= lb2 and ub1 >= ub2


def partial_overlap(cr_1: CriticalRegion, cr_2: CriticalRegion) -> bool:
    """
    Checks if the first critical region is to the left of the second critical region but overlapping.

    :param cr_1: the first critical region
    :param cr_2: the second critical region
    :return: a boolean indicating whether the first critical region is to the left of the second critical region but overlapping
    """
    # region 1 to the left of region 2
    lb1, ub1 = get_bounds_1d(cr_1.E, cr_1.f)
    lb2, ub2 = get_bounds_1d(cr_2.E, cr_2.f)
    return lb1 < lb2 and ub1 > lb2 and ub2 > ub1


def equal_linear_objective(program: MPMILP_Program, cr_1: CriticalRegion, cr_2: CriticalRegion) -> bool:
    """
    Checks if the linear objective functions of two critical regions are equal over the entire overlap.
    Since we require linear objectives, it is sufficient to check equality at two points.

    :param program: the MPMILP_Program to be solved
    :param cr_1: the first critical region
    :param cr_2: the second critical region
    :return: a boolean indicating whether the linear objective functions are equal over the entire overlap
    """

    f_inner_lb, f_inner_ub, f_outer_lb, f_outer_ub = evaluate_objective_at_overlap_bounds(program, cr_1, cr_2)
    return f_inner_lb == f_outer_lb and f_inner_ub == f_outer_ub

def region_dominates(program: MPMILP_Program, cr_1: CriticalRegion, cr_2: CriticalRegion) -> bool:
    """
    Checks if cr_1 dominates cr_2 in the sense that the objective value of cr_1 is always less than or equal to that of cr_2.

    :param program: the MPMILP_Program to be solved
    :param cr_1: the first critical region
    :param cr_2: the second critical region
    :return: a boolean indicating whether cr_1 dominates cr_2
    """
    f_1_lower, f_1_upper, f_2_lower, f_2_upper = evaluate_objective_at_overlap_bounds(program, cr_1, cr_2)
    return f_1_lower <= f_2_lower and f_1_upper <= f_2_upper

# cr_1 is the outer region or the left region, depending on calling context
def compute_objective_intersection_point(program: MPMILP_Program, cr_1: CriticalRegion, cr_2: CriticalRegion) -> Tuple[
    float, bool]:
    """
    Computes the point at which the objective functions of two critical regions intersect. Also returns a flag indicating whether the first region should be expanded to the left of the intersection point.

    :param program: the MPMILP_Program to be solved
    :param cr_1: the first critical region, which is either the outer region or the left region, depending on calling context
    :param cr_2: the second critical region, which is either the inner region or the right region, depending on calling context
    :return: a tuple of a float representing the intersection point and a boolean indicating whether the first region should be expanded to the left of the intersection point
    """
    f_1_lower, f_1_upper, f_2_lower, f_2_upper = evaluate_objective_at_overlap_bounds(program, cr_1, cr_2)
    deltaf_1 = f_1_upper - f_1_lower
    deltaf_outer = f_2_upper - f_2_lower
    lower, upper = find_overlap_bounds(cr_1, cr_2)
    delta = upper - lower
    intersection_point = (f_2_lower - f_1_lower) / (deltaf_1 - deltaf_outer) * delta + lower
    expand_left_of_intersection = f_1_lower < f_2_lower # if the objective of cr_1 is lower to the left of the intersection, we want to expand cr_1 on the left
    return intersection_point, expand_left_of_intersection


def evaluate_objective_at_overlap_bounds(program: MPMILP_Program, cr_1: CriticalRegion, cr_2: CriticalRegion) -> Tuple[
    float, float, float, float]:
    """
    Evaluates the objective function of two critical regions at the bounds of their overlap.
    
    :param program: the MPMILP_Program to be solved
    :param cr_1: the first critical region
    :param cr_2: the second critical region
    :return: a tuple of four floats representing the objective values at the bounds of the overlap
    """

    lower, upper = find_overlap_bounds(cr_1, cr_2)
    f_1_lower = program.evaluate_objective(cr_1.evaluate(numpy.array([[lower]])), numpy.array([[lower]]))
    f_1_upper = program.evaluate_objective(cr_1.evaluate(numpy.array([[upper]])), numpy.array([[upper]]))
    f_2_lower = program.evaluate_objective(cr_2.evaluate(numpy.array([[lower]])), numpy.array([[lower]]))
    f_2_upper = program.evaluate_objective(cr_2.evaluate(numpy.array([[upper]])), numpy.array([[upper]]))
    return f_1_lower, f_1_upper, f_2_lower, f_2_upper


def find_overlap_bounds(cr_1: CriticalRegion, cr_2: CriticalRegion) -> Tuple[float, float]:
    """
    Finds the space in which the regions overlap.
    If one region is fully inside the other, this will be the bounds of the inner region.
    If the regions partially overlap, this will be the lower bound of the right region and the upper bound of the left region.

    :param cr_1: the first critical region
    :param cr_2: the second critical region
    :return: a tuple of two floats representing the bounds of the overlap 
    """
    lb_1, ub_1 = get_bounds_1d(cr_1.E, cr_1.f)
    lb_2, ub_2 = get_bounds_1d(cr_2.E, cr_2.f)
    values = [lb_1, ub_1, lb_2, ub_2]
    values.sort()
    return values[1], values[2]
