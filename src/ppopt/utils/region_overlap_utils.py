from itertools import combinations, permutations
from typing import List, Tuple
from collections import deque

import numpy
from ..mpmilp_program import MPMILP_Program
from ..critical_region import CriticalRegion

from .mpqp_utils import get_bounds_1d, is_full_dimensional_1d


def reduce_overlapping_critical_regions_1d(program: MPMILP_Program, regions: List[CriticalRegion]) -> Tuple[List[CriticalRegion], bool]:
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
    while to_check:
    # for cr1, cr2 in permutations(regions, 2):
        region_added = False
        cr1, cr2 = to_check.popleft()
        # check if region has already been marked for removal, if so, no need to check again
        if cr1 in to_remove or cr2 in to_remove:
            continue
        lb1, ub1 = get_bounds_1d(cr1.E, cr1.f)
        lb2, ub2 = get_bounds_1d(cr2.E, cr2.f)

        x1s = [cr1.evaluate(numpy.array([[x]])) for x in [lb1, ub1, lb2, ub2]]
        x2s = [cr2.evaluate(numpy.array([[x]])) for x in [lb1, ub1, lb2, ub2]]

        f1ub = program.evaluate_objective(x1s[1], numpy.array([[ub1]]))
        f2lb = program.evaluate_objective(x2s[0], numpy.array([[lb2]]))
        f2ub = program.evaluate_objective(x2s[1], numpy.array([[ub2]]))
        # check if region 2 is fully inside region 1
        if full_overlap(cr1, cr2):
            # if objective of 2 is always greater than objective of 1, just discard 2
            # have evaluations of objective for CR2, need them at bounds of CR2 for CR1
            f1lb2 = program.evaluate_objective(x1s[2], numpy.array([[lb2]]))
            f1ub2 = program.evaluate_objective(x1s[3], numpy.array([[ub2]]))
            if f1lb2 == f2lb and f1ub2 == f2ub:
                # equal objective value over entire overlapping region indicates possible dual degeneracy
                # in this case, we keep both regions to have both solutions
                possible_dual_degeneracy = True
            elif f1lb2 <= f2lb and f1ub2 <= f2ub:
                to_remove = mark_for_removal(cr2, to_remove)
            elif f1lb2 > f2lb and f1ub2 > f2ub:
                # objective of 2 always less than objective of 1 --> cr1_l, cr2, cr1_r
                new_regions, cr1 = split_outer_region(new_regions, cr1, lb2, ub2)
                region_added = True
            else:
                # compute intersection point between CR1 and CR2
                deltaf1 = f1ub2 - f1lb2
                deltaf2 = f2ub - f2lb
                delta = ub2 - lb2
                intersection = (f1lb2 - f2lb) / (deltaf2 - deltaf1) * delta + lb2
                if f1lb2 > f2lb:
                    # new CRs:
                    # CR1_l = [lb1, lb2]
                    # CR2 = [lb2, intersection]
                    # CR1_r = [intersection, ub1]
                    new_regions, cr2, cr1 = adjust_regions_at_intersection(new_regions, cr2, cr1, lb2, intersection, ub1, False)
                else:
                    # new CRs:
                    # CR1_l = [lb1, intersection]
                    # CR2 = [intersection, ub2]
                    # CR1_r = [ub2, ub1]
                    new_regions, cr2, cr1 = adjust_regions_at_intersection(new_regions, cr2, cr1, lb2, intersection, ub1, True)
                region_added = True
        # check if region 1 is to the left of region 2 but overlapping
        elif partial_overlap(cr1, cr2):
            # determine lower objective value in overlap and adjust region
            if f1ub < f2lb:
                cr2 = tighten_lb(cr2, ub1)
            else:
                cr1 = tighten_ub(cr1, lb2)
        # if we added a new region, add all permutations with this region to the queue except for those with regions just checked (new region is a subset of those regions, no need to check again)
        if region_added:
            other_regions = [cr for cr in regions if cr not in [cr1, cr2]]
            to_check.extend([r, new_regions[-1]] for r in other_regions)
            to_check.extend([new_regions[-1], r] for r in other_regions)

    # add new
    regions = regions + new_regions
    # purge
    regions = [cr for cr in regions if cr not in to_remove]

    return possible_dual_degeneracy, regions


def append_region(regions: List[CriticalRegion], cr: CriticalRegion) -> List[CriticalRegion]:
    regions.append(
        CriticalRegion(cr.A, cr.b, cr.C, cr.d, cr.E, cr.f, cr.active_set, cr.omega_set, cr.lambda_set, cr.regular_set,
                       cr.y_fixation, cr.y_indices, cr.x_indices))
    return regions


def adjust_regions_at_intersection(new_regions: List[CriticalRegion], inner_region: CriticalRegion, outer_region: CriticalRegion,
                                   inner_lb: float, intersection: float, inner_ub: float,
                                   intersection_on_left: bool) -> Tuple[List[CriticalRegion], CriticalRegion, CriticalRegion]:
    new_regions = append_region(new_regions, outer_region)
    if intersection_on_left:
        outer_region = tighten_ub(outer_region, intersection)
        inner_region = tighten_lb(inner_region, intersection)
        new_regions[-1] = new_regions[-1].tighten_lb(inner_ub)
    else:
        outer_region = tighten_ub(outer_region, inner_lb)
        inner_region = tighten_ub(inner_region, intersection)
        new_regions[-1] = new_regions[-1].tighten_lb(intersection)
    return new_regions, inner_region, outer_region


def mark_for_removal(cr: CriticalRegion, removal_list: List[CriticalRegion]) -> List[CriticalRegion]:
    removal_list.append(cr)
    return removal_list


def split_outer_region(new_regions: List[CriticalRegion], cr: CriticalRegion, inner_lb: float, inner_ub: float) -> Tuple[List[CriticalRegion], CriticalRegion]:
    new_regions = append_region(new_regions, cr)
    cr.E = numpy.concatenate([cr.E, [[1]]], 0)
    cr.f = numpy.concatenate([cr.f, [[inner_lb]]])
    new_regions[-1].E = numpy.concatenate([new_regions[-1].E, [[-1]]], 0)
    new_regions[-1].f = numpy.concatenate([new_regions[-1].f, [[-inner_ub]]], 0)
    return new_regions, cr


def tighten_ub(cr: CriticalRegion, new_ub: float) -> CriticalRegion:
    cr.E = numpy.concatenate([cr.E, [[1]]], 0)
    cr.f = numpy.concatenate([cr.f, [[new_ub]]], 0)
    return cr


def tighten_lb(cr: CriticalRegion, new_lb: float) -> CriticalRegion:
    cr.E = numpy.concatenate([cr.E, [[-1]]], 0)
    cr.f = numpy.concatenate([cr.f, [[-new_lb]]], 0)
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