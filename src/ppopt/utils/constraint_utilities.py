#
# Constraint manipulation utilities
#

from typing import List, Optional, Tuple

import numpy

from ..solver_interface import solver_interface
from ..utils.general_utils import ppopt_block, select_not_in_list


# returns the norm of the constraint
def constraint_norm(A: numpy.ndarray) -> numpy.ndarray:
    """
    Finds the L2 norm of each row of a matrix

    :param A: numpy matrix.
    :return: A column vector of the row norms.
    """
    return numpy.linalg.norm(A, axis=1, keepdims=True)


# returns [A,b] scaled to mag of one
def scale_constraint(A: numpy.ndarray, b: numpy.ndarray) -> List[numpy.ndarray]:
    """
    Normalizes constraints based on the L2 norm.

    :param A: LHS Matrix constraint
    :param b: RHS column vector constraint
    :return: a list [A_scaled, b_scaled] of normalized constraints
    """
    norm_val = 1.0 / numpy.linalg.norm(A, axis=1, keepdims=True)
    return [A * norm_val, b * norm_val]


# noinspection SpellCheckingInspection,GrazieInspection
def detect_implicit_equalities(A: numpy.ndarray, b: numpy.ndarray) -> List[List[int]]:
    r"""
    Detects inequality constraints that form implicit equality constraints. This is important because inequality
    constraint pairs that form equality constraints will actively mess with the true cardinality of the active set.
    Older solvers did not make check this and that led to some problematic results.

    .. math::

        \begin{align*}
        -\langle a, x \rangle &\leq -b\\
        \langle a, x \rangle &\leq b\\
        \end{align*} \implies \langle a, x \rangle = b

    :param A: LHS of inequality set
    :param b: RHS of inequality set
    :return: List of all implicit inequality pairs
    """

    block = numpy.zeros((A.shape[0], A.shape[1] + 1))
    block[:, :A.shape[1]] = A
    block[:, A.shape[1]:A.shape[1] + 1] = b

    # scale all the constraints again, so we don't have weird numerical problems
    block = block / numpy.linalg.norm(block, axis=1, keepdims=True)

    # scale again for extra measure
    block = block / numpy.linalg.norm(block, axis=1, keepdims=True)

    implicit_pairs = []

    for i in range(block.shape[0]):
        for j in range(i, block.shape[0]):

            # we have a three tier approach to this to fight off numerical instability
            # is u ~ -v

            # First is if ||u|| = 1 then <u, v> = -1 -> <u, v>+1 ~ 0

            # second is ||u + v|| ~ 0

            # third is u_i + v_i ~ 0

            checks = 0

            # check one
            if abs(block[i].T @ block[j] + 1) <= 1e-8:
                checks += 1

            # check two
            if numpy.linalg.norm(block[i] - block[j], 2) <= 1e-12:
                checks += 1

            # check three
            if numpy.allclose(block[i], -block[j]):
                checks += 1

            if checks >= 2:
                implicit_pairs.append([i, j])

    return implicit_pairs


# removes zeroed constraints
def remove_zero_rows(A: numpy.ndarray, b: numpy.ndarray) -> List[numpy.ndarray]:
    """
    Finds rows equal to zero in A and then removes them from A and b

    :param A: LHS Matrix constraint
    :param b: RHS Column vector
    :return: a list[A_cleaned, b_cleaned] of filtered constraints
    """
    non_zero_indices = [i for i, t in enumerate(A) if numpy.nonzero(t)[0].shape[0] > 0]
    return [A[non_zero_indices], b[non_zero_indices]]


def row_equality(row_1: numpy.ndarray, row_2: numpy.ndarray, tol=10.0 ** (-16)) -> bool:
    """
    Tests if 2 row vectors are approximately equal

    :param row_1:
    :param row_2:
    :param tol: tolerable L2 norm of the difference
    :return: True if rows are equal
    """
    return numpy.sum((row_1 - row_2) ** 2) < tol


def remove_duplicate_rows(A: numpy.ndarray, b: numpy.ndarray) -> List[numpy.ndarray]:
    """Finds and removes duplicate rows in the constraints A @ x <= b."""
    combined = numpy.hstack((A, b.reshape(b.size, 1)))

    if A.size == 0 or b.size == 0:
        return [A, b]

    uniques = numpy.sort(numpy.unique(combined, axis=0, return_index=True)[1])

    return [A[uniques], b[uniques]]


def remove_duplicate_rows_mixed(A: numpy.ndarray, b: numpy.ndarray, F: numpy.ndarray) -> List[numpy.ndarray]:
    """Finds and removes duplicate rows in the constraints A @ x <= b + F @ theta."""
    combined = numpy.hstack((A, b.reshape(b.size, 1), F))

    if A.size == 0 or b.size == 0:
        return [A, b]

    uniques = numpy.sort(numpy.unique(combined, axis=0, return_index=True)[1])

    return [A[uniques], b[uniques], F[uniques]]


def facet_ball_elimination(A: numpy.ndarray, b: numpy.ndarray) -> List[numpy.ndarray]:
    """
    Removes weakly redundant constraints, method is from the appendix of the Oberdieck paper
    url: https://www.sciencedirect.com/science/article/pii/S0005109816303971

    :param A: LHS constraint matrix
    :param b: RHS constraint column vector
    :return: The processes' constraint pair [A, b]
    """
    [A_ps, b_ps] = scale_constraint(A, b)

    saved_constraints = calculate_redundant_constraints(A_ps, b_ps)

    return [A[saved_constraints], b[saved_constraints]]


def calculate_redundant_constraints(A, b):
    """
    Removes weakly redundant constraints, method is from the appendix of the Oberdieck paper
    url: https://www.sciencedirect.com/science/article/pii/S0005109816303971

    :param A: LHS constraint matrix
    :param b: RHS constraint column vector
    :return: The processes' constraint pair [A, b]
    """
    [A_ps, b_ps] = scale_constraint(A, b)

    output = []

    for i in range(A.shape[0]):

        A_norm = numpy.linalg.norm(1 - numpy.linalg.norm(A_ps @ A_ps[i].T)) * numpy.ones((A_ps.shape[0], 1))
        A_norm[i][0] = 0

        A_ball = ppopt_block([A_ps, A_norm])
        b_ball = b_ps

        obj = numpy.zeros((A_ps.shape[1] + 1, 1))
        obj[A_ps.shape[1]] = numpy.array([-1])

        soln = solver_interface.solve_lp(obj, A_ball, b_ball, [i])

        if soln is not None:
            if soln.sol[-1] > 0:
                output.append(i)

    return output


def find_redundant_constraints(A: numpy.ndarray, b: numpy.ndarray, equality_set: Optional[List[int]] = None,
                               solver='gurobi'):
    """"""
    if equality_set is None:
        equality_set = []

    redundant = []

    to_check = [x for x in range(A.shape[0]) if x not in equality_set]

    for i in to_check:
        if solver_interface.solve_lp(None, A, b, [*equality_set, i], deterministic_solver=solver) is None:
            redundant.append(i)

    return [i for i in range(A.shape[0]) if i not in redundant]


def find_redundant_constraints_with_quadratic(A: numpy.ndarray, b: numpy.ndarray, Q_q: List[numpy.ndarray],
                                               A_q: numpy.ndarray, b_q: numpy.ndarray,
                                               equality_set: Optional[List[int]], solver='gurobi'):
    """Find and remove redundant constraints from a set of linear and quadratic constraints."""
    if equality_set is None:
        equality_set = []

    redundant = []

    num_linear = A.shape[0]
    num_quadratic = len(Q_q)

    # transform single equality set to a set for linear and quadratic constraints, each starting at 0
    linear_set = [i for i in equality_set if i < num_linear]
    quadratic_set = [i-num_linear for i in equality_set if i >= num_linear]

    to_check_linear = [x for x in range(num_linear) if x not in linear_set]
    to_check_quadratic = [x for x in range(num_quadratic) if x not in quadratic_set]

    for i in to_check_linear:
        if solver_interface.solve_miqcqp(None, None, A, b, Q_q, A_q, b_q, [*linear_set, i], quadratic_set, deterministic_solver=solver, get_duals=False) is None:
            redundant.append(i)

    for i in to_check_quadratic:
        if solver_interface.solve_miqcqp(None, None, A, b, Q_q, A_q, b_q, linear_set, [*quadratic_set, i], get_duals=False, deterministic_solver=solver) is None:
            redundant.append(i + num_linear) # add the offset back so that the numbering is the same as in the original equality_set

    return [i for i in range(num_linear + num_quadratic) if i not in redundant]


def remove_strongly_redundant_constraints(A: numpy.ndarray, b: numpy.ndarray, include_kept_indices=False,
                                          deterministic_solver: str = 'gurobi'):
    """Removes strongly redundant constraints by testing the feasibility of each constraint if activated."""
    keep_list: List[int] = []
    new_index: List[int] = []
    for i in range(A.shape[0]):
        sol = solver_interface.solve_lp(None, A, b, [i], deterministic_solver=deterministic_solver)
        if sol is not None:
            keep_list.append(i)
            if len(new_index) == 0:
                new_index.append(0)
            else:
                new_index.append(new_index[-1] + 1)
    if not include_kept_indices:
        return [A[keep_list], b[keep_list]]
    else:
        return A[keep_list], b[keep_list], keep_list, new_index


def is_full_rank(A: numpy.ndarray, indices: Optional[List[int]] = None) -> bool:
    """
    Tests if the matrix A[indices] is full rank
    Empty matrices e.g. A[[]] will default to be full rank

    :param A: Matrix
    :param indices: indices to consider in rank
    :return: if the matrix is full rank or not
    """
    # consider empty matrices full rank
    if indices is None:
        return numpy.linalg.matrix_rank(A) == A.shape[0]
    if len(indices) == 0:
        return True
    return numpy.linalg.matrix_rank(A[indices]) == len(indices)


def cheap_remove_redundant_constraints(A: numpy.ndarray, b: numpy.ndarray) -> List[numpy.ndarray]:
    """
    Removes zero rows, normalizes the constraint rows to ||A_i||_{L_2} = 1, and removes duplicate rows

    :param A: LHS constraint matrix
    :param b: RHS constraint column vector
    :return: The processes' constraint pair [A, b]
    """
    # removes zeroed rows
    A, b = remove_zero_rows(A, b)

    # normalizes the rows
    A, b = scale_constraint(A, b)

    # remove duplicate rows
    A, b = remove_duplicate_rows(A, b)

    return [A, b]


def process_region_constraints(A: numpy.ndarray, b: numpy.ndarray, deterministic_solver: str = 'gurobi') -> List[
    numpy.ndarray]:
    """
    Removes all strongly and weakly redundant constraints

    :param A: LHS constraint matrix
    :param b: RHS constraint column vector
    :param deterministic_solver: the exact solver to be used
    :return: The processes' constraint pair [A, b]
    """
    A, b = cheap_remove_redundant_constraints(A, b)

    # expensive step, this solves LPs to remove all redundant constraints remaining
    A, b = remove_strongly_redundant_constraints(A, b, deterministic_solver=deterministic_solver)

    A, b = facet_ball_elimination(A, b)

    return [A, b]


def get_indices_of_zero_rows(A: numpy.ndarray, epsilon: float = 10 ** (-6)) -> Tuple[list, list]:
    is_zero = lambda x: numpy.linalg.norm(x) >= epsilon

    # sorts rows based on if they are zeros (numerically) or non-zero
    kept_constrs = [i for i, x in enumerate(A) if is_zero(x)]
    removed_constrs = [i for i, x in enumerate(A) if not is_zero(x)]

    return kept_constrs, removed_constrs


def shuffle_processed_constraints(A: numpy.ndarray, b: numpy.ndarray, F: numpy.ndarray, A_t: numpy.ndarray,
                                  b_t: numpy.ndarray, kept: list, remove: list):
    """
    An auxiliary function to shuffle constraints between the main body and parametric constraints

    :param A: The LHS constraint matrix for main body constraints
    :param b: the RHS constraint matrix for main body constraints
    :param F: the RHS parametric uncertainty matrix in the main body constraints
    :param A_t: the LHS constraint matrix for parametric constraints
    :param b_t: the RHS constraint vector for parametric constraints
    :param kept: The indices of the constraints to keep
    :param remove: The indices of the constraints to remove
    :return: The filtered constraint matrix set A, b, F, A_t, b_t
    """
    # add the purely parametric constraints to the parametric constraint set A_t, b_t
    if len(remove) > 0:
        A_t = ppopt_block([[A_t], [-F[remove]]])
        b_t = ppopt_block([[b_t], [b[remove]]])

    # remove the purly parametric constraints in the main body objective
    A = A[kept]
    b = b[kept]
    F = F[kept]

    return A, b, F, A_t, b_t


def get_independent_rows(A):
    """
    Finds the linearly independent rows of a matrix, A and returns the indices of these rows

    :param A: The matrix to find the linearly independent rows of
    :return: The indices of the linearly independent rows
    """
    num_rows = A.shape[0]
    row_ranks = numpy.zeros(num_rows)
    for i in range(num_rows - 1):
        row_ranks[i] = numpy.linalg.matrix_rank(A[:i + 1])
    rank_change = numpy.diff(row_ranks, prepend=0) > 0
    return [index for index, val in enumerate(rank_change) if val]  # select these rows as they are linear independent


def generate_reduced_equality_constraints(A, b, F, equality_indices):
    """
    Generates a reduced set of equality constraints that are linearly independent and full rank from the original set of
    equality constraints

    :param A: The LHS constraint matrix for main body constraints
    :param b: the RHS constraint vector for main body constraints
    :param F: the RHS parametric uncertainty matrix in the main body constraints
    :param equality_indices: The indices of the equality constraints
    :return: The filtered constraint matrix set A, b, F and the new equality set
    """
    # if there are no equality constraints, then we can just return the original constraints
    if len(equality_indices) == 0:
        return A, b, F, []

    # if there are equality constraints, but they are full rank, then we can just return the original constraints
    if is_full_rank(A, equality_indices):
        return A, b, F, equality_indices

    # we are in the situation where we have equality constraints that are not full rank
    new_equality_indices = get_independent_rows(A[equality_indices])

    A_eq = A[new_equality_indices]
    b_eq = b[new_equality_indices]
    F_eq = F[new_equality_indices]

    A_ineq = select_not_in_list(A, equality_indices)
    b_ineq = select_not_in_list(b, equality_indices)
    F_ineq = select_not_in_list(F, equality_indices)

    return numpy.block([[A_eq], [A_ineq]]), numpy.block([[b_eq], [b_ineq]]), numpy.block(
        [[F_eq], [F_ineq]]), new_equality_indices


def process_program_constraints(A: numpy.ndarray, b: numpy.ndarray, F: numpy.ndarray, A_t: numpy.ndarray,
                                b_t: numpy.ndarray, epsilon: float = 10 ** (-6)):
    r"""
    This is the main routine for removing redundant constraints and filtering constraints to the correct constraint set

    .. math::
        \begin{align}
        Ax &\leq b + F\theta\\
        A_{eq}x &= b_{eq}\\
        A_\theta \theta &\leq b_\theta\\
        x &\in R^n, \theta \in R^m\\
        \end{align}

    :param A: The LHS constraint matrix for main body constraints
    :param b: the RHS constraint matrix for main body constraints
    :param F: the RHS parametric uncertainty matrix in the main body constraints
    :param A_t: the LHS constraint matrix for parametric constraints
    :param b_t: the RHS constraint vector for parametric constraints
    :param epsilon: The numerical value to determine if something is a 'zero' row
    :return: The filtered constraint matrix set A, b, F, A_t, b_t
    """

    # if there are any constraints in the main constraint body A@x <= b + F@theta with ||A_i, F_i|| = 0, then this is
    # simply asking if b_i >= 0, we move this possible infeasibility to the parametric constraint set
    keep, move = get_indices_of_zero_rows(ppopt_block([[A, -F]]), epsilon)

    # move all purely 0 <= b_i constraints into the parametric constraint set
    A, b, F, A_t, b_t = shuffle_processed_constraints(A, b, F, A_t, b_t, keep, move)

    # if there are any constraints in the main constraint A@x <= b + F@theta with ||A_i|| = 0, then those can be
    # moved to the parametric constraints as it is of the form 0 <= b_i + F_i@theta
    keep, move = get_indices_of_zero_rows(A, epsilon)

    # add the purely parametric constraints to the parametric constraint set A_t, b_t
    A, b, F, A_t, b_t = shuffle_processed_constraints(A, b, F, A_t, b_t, keep, move)

    return A, b, F, A_t, b_t


def find_implicit_equalities(A: numpy.ndarray, b: numpy.ndarray, F: numpy.ndarray, equality_indices):
    r"""
    Find Implicit equalities in the main constraint block Ax <= b + F theta. E.g.  L <= A_ix - F theta <= L. Which is
    equivalent to the direct constraint A_ix = b_i + F_i theta. Also detects when c^tx - d^t theta <= b and
    c^tx - d^T theta = b.

    .. math::
        \begin{align}
        Ax &\leq b + F\theta\\
        A_{eq}x &= b_{eq}\\
        A_\theta \theta &\leq b_\theta\\
        x &\in R^n, \theta \in R^m\\
        \end{align}

    :param A: The LHS constraint matrix for main body constraints
    :param b: the RHS constraint matrix for main body constraints
    :param F: the RHS parametric uncertainty matrix in the main body constraints
    :param equality_indices: Indices of equality constraints
    :return: The filtered constraints matrix set A, b, F and the new equality set
    """
    problem_A = ppopt_block([[A, -F]])
    problem_b = ppopt_block([[b]])

    num_constraints = A.shape[0]

    constraint_pairs = detect_implicit_equalities(problem_A, problem_b)

    keep = [i[0] for i in constraint_pairs]
    remove = [i[1] for i in constraint_pairs]

    keep = list(set(keep))
    keep.sort()

    remove = list(set(remove))
    remove.sort()

    # make sure to only remove the unneeded inequalities -> only for duplicate constraints
    remove = [i for i in remove if i not in keep]

    # our temporary new active set for the problem
    temp_active_set = [*equality_indices, *keep]

    # what we are keeping
    survive = lambda x: x not in temp_active_set and x not in remove
    kept_ineqs = [i for i in range(num_constraints) if survive(i)]

    # data marshaling
    A_eq = A[temp_active_set]
    b_eq = b[temp_active_set]
    F_eq = F[temp_active_set]

    A_ineq = A[kept_ineqs]
    b_ineq = b[kept_ineqs]
    F_ineq = F[kept_ineqs]

    A = ppopt_block([[A_eq], [A_ineq]])
    b = ppopt_block([[b_eq], [b_ineq]])
    F = ppopt_block([[F_eq], [F_ineq]])

    # update problem active set
    equality_indices = list(range(len(temp_active_set)))

    return A, b, F, equality_indices


def numerically_nonzero_rows(A) -> List[int]:
    return [index for index, row in enumerate(A) if not numpy.allclose(A[index], 0, atol=10**-8)]


def remove_numerically_zero_rows(A, b) -> Tuple[numpy.ndarray, numpy.ndarray]:
    num_nonzero = numerically_nonzero_rows(A)
    keep = [i for i in range(A.shape[0]) if i in num_nonzero]
    return A[keep], b[keep]
