from typing import Iterable, Optional

import numpy

try:
    from ctypes import c_double, c_int

    import daqp
except ImportError:
    pass

from ..solver_interface.solver_interface_utils import SolverOutput
from ..utils.general_utils import make_column


def solve_qp_daqp(Q: numpy.ndarray, c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray,
                  equality_constraints: Optional[Iterable[int]] = None, verbose=False,
                  get_duals: bool = True) -> Optional[SolverOutput]:
    r"""
    Calls DAQP to solve the following optimization problem

    .. math::

        \min_{x} \frac{1}{2}x^TQx + c^Tx

    .. math::
        \begin{align}
        Ax &\leq b\\
        A_{eq}x &= b_{eq}\\
        x &\in R^n\\
        \end{align}


    :param Q: Square matrix
    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints, can be None
    :param verbose: Flag for output of underlying Solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

    :return: A SolverOutput object if optima found, otherwise None.
    """

    if equality_constraints is None:
        equality_constraints = []

    num_constraints = A.shape[0]
    num_x = A.shape[1]
    num_equality_constraints = len(equality_constraints)
    num_inequality_constraints = num_constraints - num_equality_constraints

    if c is None:
        c = numpy.zeros(num_x).reshape(-1, 1)

    if A is None or b is None:
        # this is an unconstrained problem thus we can solve the unconstrained problem in numpy
        # we should likely just make this a function
        x_sol = numpy.linalg.solve(Q, -c).reshape(-1, 1)
        opt_val = float((0.5 * x_sol.T @ Q @ x_sol + c.T @ x_sol)[0, 0])
        return SolverOutput(opt_val, x_sol, numpy.array([]), numpy.array([]), numpy.array([]))

    ineq = [i for i in range(A.shape[0]) if i not in equality_constraints]

    if num_equality_constraints != 0:
        new_A = A[[*equality_constraints, *ineq]].astype(c_double)
        new_b = b[[*equality_constraints, *ineq]].flatten().astype(c_double)
        # equality constraints are labeled as 5, regular inequalities are 0
        constraint_sense = numpy.array([*[5 for _ in range(num_equality_constraints)],
                                        *[0 for _ in range(num_inequality_constraints)]]).astype(c_int)
    else:
        new_A = A.astype(c_double)
        new_b = b.flatten().astype(c_double)
        constraint_sense = numpy.array([0 for _ in range(num_inequality_constraints)]).astype(c_int)

    Q_ = (0.5 * (Q + Q.T)).astype(c_double)
    c_ = c.flatten().astype(c_double)
    blower = numpy.full(num_inequality_constraints+num_equality_constraints, -1e30)
    x_star, opt, status, info = daqp.solve(H=Q_, f=c_, A=new_A, bupper=new_b, blower=blower, sense=constraint_sense)

    # if there is anything other than an optimal solution found return nothing
    if status != 1:
        return None

    duals = info['lam']
    lagrange = numpy.zeros(num_constraints)
    indexing = [*equality_constraints, *ineq]
    lagrange[indexing] = duals
    lagrange[ineq] = -lagrange[ineq]

    slack = b - A @ make_column(x_star)
    active = []

    for i in range(num_constraints):
        if abs(slack[i]) <= 10 ** -10 or lagrange[i] != 0:
            active.append(i)

    return SolverOutput(opt, x_star, slack.flatten(), numpy.array(active).astype('int64'), lagrange)
