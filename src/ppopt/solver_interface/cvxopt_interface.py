from typing import Optional

import numpy

try:
    from cvxopt import matrix, solvers
except ImportError:
    pass

from ..solver_interface.solver_interface_utils import SolverOutput, get_program_parameters


def process_cvxopt_solution(sol, equality_constraints, inequality_constraints, num_constraints, get_duals) -> Optional[
    SolverOutput]:
    status = sol['status']

    if status != 'optimal':
        return None

    slack = numpy.zeros(num_constraints)
    lagrange = numpy.zeros(num_constraints)

    slack[equality_constraints] = 0
    slack[inequality_constraints] = numpy.array(sol['s']).flatten()

    if len(equality_constraints) > 0:
        lagrange[equality_constraints] = -numpy.array(sol['y']).flatten()
    lagrange[inequality_constraints] = -numpy.array(sol['z']).flatten()

    active = []

    for i in range(num_constraints):
        if abs(slack[i]) <= 10 ** -10 or lagrange[i] != 0:
            active.append(i)

    active = numpy.array(active)

    if not get_duals:
        lagrange = None

    return SolverOutput(obj=sol['primal objective'], sol=numpy.array(sol['x']).flatten(), slack=slack,
                        active_set=active,
                        dual=lagrange)


def separate_constraints(A, b, equality_constraints, ineq):
    """

    :param A:
    :param b:
    :param equality_constraints:
    :param ineq:
    :return:
    """
    if A is None or b is None:
        return None, None, None, None

    A_eq = matrix(A[equality_constraints])
    b_eq = matrix(b[equality_constraints])
    A_ineq = matrix(A[ineq])
    b_ineq = matrix(b[ineq])

    if len(ineq) == 0:
        A_ineq = None
        b_ineq = None

    if len(equality_constraints) == 0:
        A_eq = None
        b_eq = None

    return A_eq, b_eq, A_ineq, b_ineq


def solve_fully_constraints(c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray) -> Optional[SolverOutput]:
    """
    This solves a fully constrained linear system by directly solving the linear system. This is NOT complete!

    :param c: Column Vector
    :param A: Constraint LHS matrix
    :param b: Constraint RHS matrix
    :return: A SolverOutput Object
    """
    x = numpy.linalg.solve(A, b)
    dual = numpy.linalg.solve(A.T, -c)
    num_vars = A.shape[0]

    sol = SolverOutput(sol=x, obj=c.T @ x, slack=numpy.zeros(num_vars),
                       active_set=numpy.array([i for i in range(num_vars)]), dual=dual)

    # check if the system agrees with the equality constraints

    if numpy.all(A @ sol.sol - b > -10 ** 8):
        return sol
    else:
        return None


def solve_qp_cvxopt(Q: numpy.ndarray, c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, equality_constraints=None,
                    verbose=False,
                    get_duals=True, cvx_solver='quadprog') -> Optional[SolverOutput]:
    r"""
    This is the breakout for QP with cvxopt

    .. math::

        \min_{x} \frac{1}{2}x^TQx + c^Tx

    .. math::
        \begin{align}
        Ax &\leq b\\
        A_{eq}x &= b_{eq}\\
        x &\in R^n\\
        \end{align}


    :param Q:
    :param c:
    :param A:
    :param b:
    :param equality_constraints:
    :param verbose:
    :param get_duals:
    :param cvx_solver:
    :return:
    """
    # this is to deal with immutability a copy constrictors
    if equality_constraints is None:
        equality_constraints = []

    num_variables, num_constraints = get_program_parameters(Q, c, A, b)

    # if n objective is supplied, set to Obj(x) = 0 aka. we are looking for feasibility
    if c is None:
        c = numpy.zeros((num_variables, 1))

    # generate list of all indices of inequality constraints
    ineq = [i for i in range(num_constraints) if i not in equality_constraints]

    A_eq, b_eq, A_ineq, b_ineq = separate_constraints(A, b, equality_constraints, ineq)

    sol = solvers.qp(P=matrix(Q), q=matrix(c), G=A_ineq, h=b_ineq, A=A_eq, b=b_eq, solver=cvx_solver)

    return process_cvxopt_solution(sol, equality_constraints, ineq, num_constraints, get_duals)


def solve_lp_cvxopt(c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, equality_constraints=None, verbose=False,
                    get_duals=True, cvx_solver='glpk') -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving linear programs with cvxopt, This is the preferred Solver as it has the lowest
    interface cost of all the other solvers

    .. math::

        \min_{x} c^Tx

    .. math::
        \begin{align}
        Ax &\leq b\\
        A_{eq}x &= b_{eq}\\
        x &\in R^n\\
        \end{align}

    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param verbose: Flag for output of underlying Solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True
    :param cvx_solver: selects the cvxopt Solver, default is glpk
    :return: A SolverOutput Object
    """
    # this is to deal with immutability a copy constrictors
    if equality_constraints is None:
        equality_constraints = []

    num_variables, num_constraints = get_program_parameters(None, c, A, b)

    # if no matrix is supplied then LP is unbounded
    if A is None:
        return None

    if A.shape[0] == 0 or A.shape[1] == 0:
        return None

    # if n objective is supplied, set to Obj(x) = 0 aka. we are looking for feasibility
    if c is None:
        c = numpy.zeros((num_variables, 1))

    # in general cvxopt fails for fully constrained systems so solve it on the side
    if len(equality_constraints) == A.shape[0]:
        return solve_fully_constraints(c, A, b)

    # generate list of all indices of inequality constraints
    ineq = [i for i in range(A.shape[0]) if i not in equality_constraints]

    A_eq, b_eq, A_ineq, b_ineq = separate_constraints(A, b, equality_constraints, ineq)

    sol = solvers.lp(c=matrix(c), G=A_ineq, h=b_ineq, A=A_eq, b=b_eq, solver=cvx_solver,
                     options={'glpk': {'msg_lev': 'GLP_MSG_OFF', 'it_lim': 1000000}})

    return process_cvxopt_solution(sol, equality_constraints, ineq, num_constraints, get_duals)
