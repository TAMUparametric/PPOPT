from typing import Optional, Iterable

import numpy

from .solver_interface_utils import SolverOutput
from ..solver_interface.cvxopt_interface import solve_lp_cvxopt
from ..solver_interface.gurobi_solver_interface import solve_qp_gurobi, solve_lp_gurobi, solve_milp_gurobi, \
    solve_miqp_gurobi
from ..solver_interface.quad_prog_interface import solve_qp_quadprog

Matrix = Optional[numpy.ndarray]


def solver_not_supported(solver_name: str) -> None:
    """This is an internal method that throws an error and prompts the user when they use an unsupported Solver"""
    supported_solvers = ['gurobi', 'cplex', 'glpk']

    message = f"Solver {solver_name} is not supported! \n" \
              + f'PPOPT Supports the following solvers {str(supported_solvers)} \n'
    raise RuntimeError(message)


# noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
def solve_miqp(Q: Matrix, c: Matrix, A: Matrix, b: Matrix,
               equality_constraints: Iterable[int] = None,
               bin_vars: Iterable[int] = None, verbose: bool = False,
               get_duals: bool = True, deterministic_solver='gurobi') -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving mixed integer quadratic programs

    .. math::

        \min_{xy} \frac{1}{2} [xy]^TQ[xy] + c^T[xy]

    .. math::
        \begin{align}
        A[xy] &\leq b\\
        A_{eq}[xy] &= b_{eq}\\
        x &\in R^n\\
        y &\in \{0, 1\}^m
        \end{align}

    :param Q: Square matrix, can be None
    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param bin_vars: List of binary variable indices
    :param verbose: Flag for output of underlying Solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)
    :param deterministic_solver: The underlying Solver to use, e.g. gurobi, ect

    :return: A SolverOutput object if optima found, otherwise None.
    """
    if deterministic_solver == "gurobi":
        return solve_miqp_gurobi(Q, c, A, b, equality_constraints, bin_vars, verbose, get_duals)
    else:
        solver_not_supported(deterministic_solver)


def solve_qp(Q: Matrix, c: Matrix, A: Matrix, b: Matrix, equality_constraints: Iterable[int] = None,
             verbose=False,
             get_duals=True, deterministic_solver='gurobi') -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving quadratic programs

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
    :param deterministic_solver: The underlying Solver to use, e.g. gurobi, ect

    :return: A SolverOutput object if optima found, otherwise None.
    """
    if deterministic_solver == "gurobi":
        return solve_qp_gurobi(Q, c, A, b, equality_constraints, verbose, get_duals)
    elif deterministic_solver == "quadprog":
        return solve_qp_quadprog(Q, c, A, b, equality_constraints, verbose, get_duals)
    else:
        solver_not_supported(deterministic_solver)


# noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
def solve_lp(c: Matrix, A: Matrix, b: Matrix, equality_constraints=None, verbose=False,
             get_duals=True, deterministic_solver='gurobi') -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving linear programs

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
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)
    :param deterministic_solver: The underlying Solver to use, e.g. gurobi, ect

    :return: A SolverOutput object if optima found, otherwise None.
    """
    if deterministic_solver == "gurobi":
        return solve_lp_gurobi(c, A, b, equality_constraints, verbose, get_duals)
    if deterministic_solver == 'glpk':
        return solve_lp_cvxopt(c, A, b, equality_constraints, verbose, get_duals)
    else:
        solver_not_supported(deterministic_solver)


def solve_milp(c: Matrix, A: Matrix, b: Matrix, equality_constraints: Iterable[int] = None,
               bin_vars: Iterable[int] = None, verbose=False, get_duals=True,
               deterministic_solver='gurobi') -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving mixed integer linear programs

    .. math::

        \min_{xy} c^T[xy]

    .. math::

        \begin{align}
        A[xy] &\leq b\\
        A_{eq}[xy] &= b_{eq}\\
        x &\in R^n\\
        y &\in \{0, 1\}^m
        \end{align}

    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param bin_vars: List of binary variable indices
    :param verbose: Flag for output of underlying Solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)
    :param deterministic_solver: The underlying Solver to use, e.g. gurobi, ect

    :return: A dictionary of the Solver outputs, or none if infeasible or unbounded. output['sol'] = primal
    variables, output['dual'] = dual variables, output['obj'] = objective value, output['const'] = slacks,
    output['active'] = active constraints.
    """
    if deterministic_solver == "gurobi":
        return solve_milp_gurobi(c, A, b, equality_constraints, bin_vars, verbose, get_duals)
    else:
        solver_not_supported(deterministic_solver)
