from typing import Optional, Iterable

import numpy

try:
    import quadprog
except ImportError:
    pass

from ..solver_interface.solver_interface_utils import SolverOutput
from ..utils.general_utils import make_column


def solve_qp_quadprog(Q: numpy.ndarray, c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray,
                      equality_constraints: Iterable[int] = None, verbose=False,
                      get_duals: bool = True) -> Optional[SolverOutput]:
    r"""
    Calls Quadprog to solve the following optimization problem

    .. math::

        \min_{x} \frac{1}{2}x^TQx + c^Tx

    .. math::
        \begin{align}
        Ax &\leq b\\
        A_{eq}x &= b_{eq}\\
        x &\in R^n\\
        \end{align}


    :param Q: Square matrix, can be None
    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param verbose: Flag for output of underlying Solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

    :return: A SolverOutput object if optima found, otherwise None.
    """
    try:
        if equality_constraints is None:
            equality_constraints = []

        num_constraints = A.shape[0]
        num_equality_constraints = len(equality_constraints)
        ineq = [i for i in range(A.shape[0]) if i not in equality_constraints]

        if num_equality_constraints != 0:
            new_A = A[[*equality_constraints, *ineq]]
            new_b = b[[*equality_constraints, *ineq]]
        else:
            new_A = A
            new_b = b

        Q_ = .5 * (Q + Q.T)

        sol = quadprog.solve_qp(G=Q_, a=-c.flatten(), C=-new_A.T, b=-new_b.flatten(), meq=len(equality_constraints))

        if sol is None:
            return None

        lagrange = numpy.zeros(num_constraints)
        x_star = sol[0]
        opt = sol[1]
        duals = sol[4]
        active = []
        indexing = [*equality_constraints, *ineq]
        lagrange[indexing] = duals
        lagrange[ineq] = -lagrange[ineq]

        slack = b - A @ make_column(x_star)

        for i in range(num_constraints):
            if abs(slack[i]) <= 10 ** -10 or lagrange[i] != 0:
                active.append(i)

        return SolverOutput(opt, x_star, slack.flatten(), numpy.array(active).astype('int64'), lagrange)

    except ValueError as _:
        # just swallow the error as something happened Infeasibility or non-symmetry
        return None
