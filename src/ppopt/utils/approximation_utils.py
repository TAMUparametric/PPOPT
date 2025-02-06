from ..nonlinear_critical_region import NonlinearCriticalRegion

import numpy

from typing import List

def point_is_relevant(program, theta_point: numpy.ndarray, critical_regions: List[NonlinearCriticalRegion], options) -> bool:
    """
    Check if a linearization point is relevant given the current critical regions
    This is the case if the point is in one of the regions and does not satisfy tolerances

    :param program: MPQCQP_Program to be solved
    :param theta_point: the point in the uncertainty space
    :param critical_regions: the critical regions
    :param options: the approximation options
    :return: True if the point is relevant, False otherwise
    """
    constraint_tol_satisfied = False
    solution_tol_satisfied = False
    is_inside_current_regions = False
    for region in critical_regions:
        if region.is_inside(theta_point, 1e-6):
            is_inside_current_regions = True
            x_region = numpy.array(region.x_star_numpy(theta_point)).reshape(-1,1)
            # check constraint tolerance
            qvals = [q.evaluate(x_region, theta_point) for q in program.qconstraints]
            if numpy.all([qval <= options.constraint_tol for qval in qvals]):
                constraint_tol_satisfied = True
            # check solution tolerance
            sol_exact = program.solve_theta(theta_point)
            if sol_exact is None:
                # then we are infeasible which means only constraint tol is relevant
                solution_tol_satisfied = True
            else:
                x_exact = sol_exact.sol.reshape(-1, 1)
                if numpy.linalg.norm(x_exact - x_region, numpy.inf) <= options.solution_tol:
                    solution_tol_satisfied = True
            break
    if not is_inside_current_regions and len(critical_regions) > 0:
        return False
    if constraint_tol_satisfied and solution_tol_satisfied:
        return False
    return True