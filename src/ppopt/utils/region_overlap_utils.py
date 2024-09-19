from itertools import combinations, permutations

import numpy
from ..mpmilp_program import MPMILP_Program
from ..critical_region import CriticalRegion

from .mpqp_utils import get_bounds_1d, is_full_dimensional_1d


def reduce_overlapping_critical_regions_1d(program: MPMILP_Program, regions: list) -> list:
    """
    This adjusts critical regions for the 1d-parameter case, so that there are no overlaps.

    :param program: the MPMILP_Program to be solved
    :param regions: a list of CriticalRegion objects
    :return: a list of non-overlapping CriticalRegion objects
    """

    # guard around the statement
    if program.num_t() != 1:
        raise ValueError('reduce_overlapping_critical_regions_1d  requires a 1d-parameter problem')

    overlaps_remaining, region_added, regions = identify_overlaps(program, regions)
    while region_added:
        possible_dual_degeneracy, region_added, regions = identify_overlaps(program, regions)
        overlaps_remaining = overlaps_remaining or possible_dual_degeneracy

    return regions, overlaps_remaining


def identify_overlaps(program, regions):
    # In a first step, we identify regions that are fully or partially overlapping If a region is fully contained in
    # another one, we make it infeasible, in order to prevent weird stuff from happening by deleting during iteration
    # In the second step, we delete all infeasible regions (i.e. keep only full dimensional regions) and add all
    # those that were newly identified
    region_added = False
    new_regions = []
    possible_dual_degeneracy = False
    # test all permutations rather than combinations: this way we only need to test for 2 cases (CR 2 fully inside CR 1, or CR 1 left of CR 2 but overlapping), since the other 2 cases are handled by the swapped permutation
    for cr1, cr2 in permutations(regions, 2):
        lb1, ub1 = get_bounds_1d(cr1.E, cr1.f)
        lb2, ub2 = get_bounds_1d(cr2.E, cr2.f)

        x1s = [cr1.evaluate(numpy.array([[x]])) for x in [lb1, ub1, lb2, ub2]]
        x2s = [cr2.evaluate(numpy.array([[x]])) for x in [lb1, ub1, lb2, ub2]]

        f1lb = program.evaluate_objective(x1s[0], numpy.array([[lb1]]))
        f1ub = program.evaluate_objective(x1s[1], numpy.array([[ub1]]))
        f2lb = program.evaluate_objective(x2s[0], numpy.array([[lb2]]))
        f2ub = program.evaluate_objective(x2s[1], numpy.array([[ub2]]))
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
                make_infeasible(cr2)
            elif f1lb2 > f2lb and f1ub2 > f2ub:
                # objective of 2 always less than objective of 1 --> cr1_l, cr2, cr1_r
                split_outer_region(new_regions, cr1, lb2, ub2)
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
                    adjust_regions_at_intersection(new_regions, cr2, cr1, lb2, intersection, ub1, False)
                else:
                    # new CRs:
                    # CR1_l = [lb1, intersection]
                    # CR2 = [intersection, ub2]
                    # CR1_r = [ub2, ub1]
                    adjust_regions_at_intersection(new_regions, cr2, cr1, lb2, intersection, ub1, True)
                region_added = True
        elif partial_overlap(cr1, cr2):
            # determine lower objective value in overlap and adjust region
            if f1ub < f2lb:
                tighten_ub(cr2, ub1)
            else:
                tighten_lb(cr1, lb2)

    # purge
    regions = [cr for cr in regions if is_full_dimensional_1d(cr.E, cr.f)]
    # add new
    regions = regions + new_regions

    return possible_dual_degeneracy, region_added, regions


def append_region(regions, cr):
    regions.append(
        CriticalRegion(cr.A, cr.b, cr.C, cr.d, cr.E, cr.f, cr.active_set, cr.omega_set, cr.lambda_set, cr.regular_set,
                       cr.y_fixation, cr.y_indices, cr.x_indices))
    return regions


def adjust_regions_at_intersection(new_regions, inner_region, outer_region, inner_lb, intersection, inner_ub,
                                   intersection_on_left):
    if intersection_on_left:
        new_regions = append_region(new_regions, outer_region)
        outer_region.E = numpy.concatenate([outer_region.E, [[1]]], 0)
        outer_region.f = numpy.concatenate([outer_region.f, [[intersection]]])
        inner_region.E = numpy.concatenate([inner_region.E, [[-1]]], 0)
        inner_region.f = numpy.concatenate([inner_region.f, [[-intersection]]])
        new_regions[-1].E = numpy.concatenate([new_regions[-1].E, [[-1]]], 0)
        new_regions[-1].f = numpy.concatenate([new_regions[-1].f, [[-inner_ub]]], 0)
    else:
        new_regions = append_region(new_regions, outer_region)
        outer_region.E = numpy.concatenate([outer_region.E, [[1]]], 0)
        outer_region.f = numpy.concatenate([outer_region.f, [[inner_lb]]])
        inner_region.E = numpy.concatenate([inner_region.E, [[1]]], 0)
        inner_region.f = numpy.concatenate([inner_region.f, [[intersection]]])
        new_regions[-1].E = numpy.concatenate([new_regions[-1].E, [[-1]]], 0)
        new_regions[-1].f = numpy.concatenate([new_regions[-1].f, [[-intersection]]], 0)


def make_infeasible(cr):
    cr.E = numpy.array([[1], [-1]])
    cr.f = numpy.array([[-1], [-1]])


def split_outer_region(new_regions, cr, inner_lb, inner_ub):
    new_regions = append_region(new_regions, cr)
    cr.E = numpy.concatenate([cr.E, [[1]]], 0)
    cr.f = numpy.concatenate([cr.f, [[inner_lb]]])
    new_regions[-1].E = numpy.concatenate([new_regions[-1].E, [[-1]]], 0)
    new_regions[-1].f = numpy.concatenate([new_regions[-1].f, [[-inner_ub]]], 0)


def tighten_lb(cr, new_lb):
    cr.E = numpy.concatenate([cr.E, [[1]]], 0)
    cr.f = numpy.concatenate([cr.f, [[new_lb]]], 0)


def tighten_ub(cr, new_ub):
    cr.E = numpy.concatenate([cr.E, [[-1]]], 0)
    cr.f = numpy.concatenate([cr.f, [[-new_ub]]], 0)

def full_overlap(cr1, cr2):
    # region 2 fully inside region 1
    lb1, ub1 = get_bounds_1d(cr1.E, cr1.f)
    lb2, ub2 = get_bounds_1d(cr2.E, cr2.f)
    return lb1 <= lb2 and ub1 >= ub2

def partial_overlap(cr1, cr2):
    # region 1 to the left of region 2
    lb1, ub1 = get_bounds_1d(cr1.E, cr1.f)
    lb2, ub2 = get_bounds_1d(cr2.E, cr2.f)
    return lb1 < lb2 and ub1 > lb2 and ub2 > ub1