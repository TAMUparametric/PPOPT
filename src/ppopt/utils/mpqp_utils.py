from typing import List, Optional

import numpy

from ..critical_region import CriticalRegion
from ..mpqp_program import MPQP_Program
from ..solver import Solver
from ..utils.constraint_utilities import (
    cheap_remove_redundant_constraints,
    remove_duplicate_rows,
    scale_constraint,
)
from ..utils.general_utils import ppopt_block
from .chebyshev_ball import chebyshev_ball


def get_boundary_types(region: numpy.ndarray, omega: numpy.ndarray, lagrange: numpy.ndarray, regular: numpy.ndarray) -> \
        List:
    """
    Classifies the boundaries of a polytope into Omega constraints, Lagrange multiplier = 0 constraints, and Activated program constraints

    :param region:
    :param omega:
    :param lagrange:
    :param regular:
    :return:
    """

    num_constraints = region.shape[0]

    is_labeled = numpy.zeros(num_constraints)

    def label(compare):
        output = []
        output_2 = []
        for i in range(num_constraints):
            for j in range(compare.shape[0]):
                if is_labeled[i] == 0:
                    if numpy.allclose(region[i], compare[j]):
                        is_labeled[i] = 1
                        output.append(i)
                        output_2.append(j)
        return output, output_2

    omega_list = label(omega)
    lagrange_list = label(lagrange)
    regular_list = label(regular)

    return [omega_list, lagrange_list, regular_list]


def build_suboptimal_critical_region(program: MPQP_Program, active_set: List[int]):
    """
    Builds the critical region without considering culling facets or any other operation.
    Primary uses for this is based on culling lower dimensional feasible sets.

    :param program: the MQMP_Program to be solved
    :param active_set: the active set combination to build this critical region from
    :return: Returns the associated critical region if fully dimensional else returns None
    """
    inactive = [i for i in range(program.num_constraints()) if i not in active_set]

    parameter_A, parameter_b, lagrange_A, lagrange_b = program.optimal_control_law(active_set)

    # reduced constraints
    omega_A, omega_b = program.A_t, program.b_t
    lambda_A, lambda_b = cheap_remove_redundant_constraints(-lagrange_A, lagrange_b)

    # x as a function of theta representation of the inactive constraints
    inactive_A = program.A[inactive] @ parameter_A - program.F[inactive]
    inactive_b = program.b[inactive] - program.A[inactive] @ parameter_b

    # reduce these constraints
    inactive_A, inactive_b = cheap_remove_redundant_constraints(inactive_A, inactive_b)

    constraints_A = ppopt_block([[omega_A], [lambda_A], [inactive_A]])
    constraints_b = ppopt_block([[omega_b], [lambda_b], [inactive_b]])

    # combine them together
    region_A, region_b = remove_duplicate_rows(constraints_A, constraints_b)

    return region_A, region_b


# noinspection PyUnusedLocal
def gen_cr_from_active_set(program: MPQP_Program, active_set: List[int], check_full_dim=True) -> Optional[
    CriticalRegion]:
    """
    Builds the critical region of the given mpqp from the active set.

    :param program: the MQMP_Program to be solved
    :param active_set: the active set combination to build this critical region from
    :param check_full_dim: Keyword Arg, if true will return null if the region has lower dimensionality
    :return: Returns the associated critical region if fully dimensional else returns None
    """

    num_equality = program.num_equality_constraints()

    active = active_set[num_equality:]
    inactive = [i for i in range(program.num_constraints()) if i not in active_set]

    parameter_A, parameter_b, lagrange_A, lagrange_b = program.optimal_control_law(active_set)

    # lagrange constraints
    lambda_A, lambda_b = -lagrange_A[num_equality:], lagrange_b[num_equality:]

    # Theta Constraints
    omega_A, omega_b = program.A_t, program.b_t

    # Inactive Constraints remain inactive
    inactive_A = program.A[inactive] @ parameter_A - program.F[inactive]
    inactive_b = program.b[inactive] - program.A[inactive] @ parameter_b

    # we need to check for zero rows
    lamba_nonzeros = [i for i, t in enumerate(lambda_A) if numpy.nonzero(t)[0].shape[0] > 0]
    ineq_nonzeros = [i for i, t in enumerate(inactive_A) if numpy.nonzero(t)[0].shape[0] > 0]

    # Block of all critical region constraints

    active = [active[indx] for indx in lamba_nonzeros]
    lambda_A = lambda_A[lamba_nonzeros]
    lambda_b = lambda_b[lamba_nonzeros]

    inactive = [inactive[indx] for indx in ineq_nonzeros]
    inactive_A = inactive_A[ineq_nonzeros]
    inactive_b = inactive_b[ineq_nonzeros]

    CR_As = ppopt_block([[lambda_A], [inactive_A], [omega_A]])
    CR_bs = ppopt_block([[lambda_b], [inactive_b], [omega_b]])

    CR_As, CR_bs = scale_constraint(CR_As, CR_bs)


    # if check_full_dim is set check if region is lower dimensional if so return None
    if check_full_dim:
        # if the resulting system is not fully dimensional return None
        if not is_full_dimensional(CR_As, CR_bs, program.solver):
            return None

    # if it is fully dimensional we get to classify the constraints and then reduce them (important)!

    kept_lambda_indices = []
    kept_inequality_indices = []
    kept_omega_indices = []

    # iterate over the non-zero lagrange constraints
    for index in range(len(lamba_nonzeros)):

        sol = program.solver.solve_lp(None, CR_As, CR_bs, [index])

        if sol is not None:
            kept_lambda_indices.append(index)

    # iterate over the non-zero inequaltity constraints
    for index in range(len(ineq_nonzeros)):

        sol = program.solver.solve_lp(None, CR_As, CR_bs, [index + len(lamba_nonzeros)])

        if sol is not None:
            kept_inequality_indices.append(index)

    # iterate over the omega constraints
    for index in range(omega_A.shape[0]):

        sol = program.solver.solve_lp(None, CR_As, CR_bs, [index + len(lamba_nonzeros) + len(ineq_nonzeros)])

        if sol is not None:
            kept_omega_indices.append(index)

    # create out reduced Critical region constraint block
    CR_As = ppopt_block(
        [[lambda_A[kept_lambda_indices]], [inactive_A[kept_inequality_indices]], [omega_A[kept_omega_indices]]])
    CR_bs = ppopt_block(
        [[lambda_b[kept_lambda_indices]], [inactive_b[kept_inequality_indices]], [omega_b[kept_omega_indices]]])


    # recover the lambda boundaries that remain
    relevant_lambda = [active[index] for index in kept_lambda_indices]

    real_regular = [inactive[index] for index in kept_inequality_indices]
    regular = [kept_inequality_indices, real_regular]

    # remove any possible duplicate constraints
    # and rescale since we did not rescale this particular set of constraints!!!
    CR_As, CR_bs = remove_duplicate_rows(CR_As, CR_bs)
    CR_As, CR_bs = scale_constraint(CR_As, CR_bs)

    return CriticalRegion(parameter_A, parameter_b, lagrange_A, lagrange_b, CR_As, CR_bs, active_set,
                          kept_omega_indices, relevant_lambda, regular)


def is_full_dimensional(A, b, solver: Solver = None):
    """
    This checks the dimensionality of a polytope defined by P = {x: Ax≤b}. Current method is based on checking if the
    radii of the chebychev ball is nonzero. However, this is numerically not so stable, and will eventually be replaced
    by looking at the ratio of the 2 chebychev balls

    :param A: LHS of polytope constraints
    :param b: RHS of polytope constraints
    :param solver: the solver interface to direct the deterministic solver
    :return: True if polytope is fully dimensional else False
    """

    if solver is None:
        solver = Solver()

    # TODO: Add second chebychev ball to get a more accurate estimate of lower dimensionality

    soln = chebyshev_ball(A, b, deterministic_solver=solver.solvers['lp'])

    if soln is not None:
        return soln.sol[-1] > 10 ** -8
    return False
