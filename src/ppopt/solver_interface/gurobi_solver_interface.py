from typing import Iterable, Optional

import numpy

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from ..solver_interface.solver_interface_utils import SolverOutput, get_program_parameters


def solve_miqp_gurobi(Q: numpy.ndarray = None, c: numpy.ndarray = None, A: numpy.ndarray = None,
                      b: numpy.ndarray = None,
                      equality_constraints: Iterable[int] = None,
                      bin_vars: Iterable[int] = None, verbose: bool = False,
                      get_duals: bool = True) -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving mixed integer quadratic programs with gruobi

    The Mixed Integer Quadratic program programming problem

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

    :return: A dictionary of the Solver outputs, or none if infeasible or unbounded. \\n output['sol'] = primal
    variables, output['dual'] = dual variables, output['obj'] = objective value, output['const'] = slacks,
    output['active'] = active constraints.
    """
    model = gp.Model()

    if not verbose:
        model.setParam("OutputFlag", 0)

    if equality_constraints is None:
        equality_constraints = []

    if bin_vars is None:
        bin_vars = []

    if len(bin_vars) == 0:
        model.setParam("Method", 0)

    if len(bin_vars) == 0 and Q is None:
        model.setParam("Method", 0)
        model.setParam("Quad", 0)

    # in the case of non-convex QPs add the non-convex flag, set the MIP gap the 0 we want exact solutions
    if Q is not None:
        if numpy.min(numpy.linalg.eigvalsh(Q)) < 0:
            model.Params.NonConvex = 2
            # noinspection SpellCheckingInspection
            model.Params.MIPgap = 0

    # define num variables and num constraints variables
    num_vars, num_constraints = get_program_parameters(Q, c, A, b)

    if A is None and Q is None:
        return None

    var_types = [GRB.BINARY if i in bin_vars else GRB.CONTINUOUS for i in range(num_vars)]
    # noinspection PyTypeChecker
    x = model.addMVar(num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=var_types)

    if num_constraints != 0:
        # sense = numpy.chararray(num_constraints)
        sense = [GRB.LESS_EQUAL for _ in range(num_constraints)]
        for i in equality_constraints:
            sense[i] = GRB.EQUAL

        # sense.fill(GRB.LESS_EQUAL)
        # sense[equality_constraints] = GRB.EQUAL
        # inequality = [i for i in range(num_constraints) if i not in equality_constraints]
        # sense[inequality] = GRB.LESS_EQUAL

        model.addMConstr(A, x, sense, b)

    objective = 0

    if Q is not None and c is None:
        objective = .5 * (x @ Q @ x)

    if c is not None and Q is None:
        objective = c.flatten() @ x

    if Q is not None and c is not None:
        objective = .5 * (x @ Q @ x) + c.flatten() @ x

    model.setObjective(objective, sense=GRB.MINIMIZE)

    model.optimize()
    model.update()

    # get gurobi status
    status = model.status
    # if not solved return None
    if status != GRB.OPTIMAL and status != GRB.SUBOPTIMAL:
        return None

    # create the Solver return object
    sol = SolverOutput(obj=model.getAttr("ObjVal"), sol=numpy.array(x.X), slack=None,
                       active_set=None, dual=None)

    # if we have a constrained system we need to add in the slack variables and active set
    if num_constraints != 0:

        if get_duals:
            # dual variables only really make sense if the system doesn't have binaries
            if len(bin_vars) == 0:
                sol.dual = numpy.array(model.getAttr("Pi"))

        sol.slack = numpy.array(model.getAttr("Slack"))
        sol.active_set = numpy.where((A @ sol.sol.flatten() - b.flatten()) ** 2 < 10 ** -12)[0]

    return sol


def solve_qp_gurobi(Q: numpy.ndarray, c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray,
                    equality_constraints: Iterable[int] = None,
                    verbose=False,
                    get_duals=True) -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving quadratic programs with gruobi

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

    :return:  A SolverOutput Object
    """
    return solve_miqp_gurobi(Q=Q, c=c, A=A, b=b, equality_constraints=equality_constraints, verbose=verbose,
                             get_duals=get_duals)


# noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
def solve_lp_gurobi(c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, equality_constraints: Iterable[int] = None,
                    verbose: bool = False,
                    get_duals: bool = True) -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving linear programs with gruobi.

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

    :return: A SolverOutput Object
    """
    if not gurobi_pretest(A, b):
        return None

    return solve_miqp_gurobi(c=c, A=A, b=b, equality_constraints=equality_constraints, verbose=verbose,
                             get_duals=get_duals)


def solve_milp_gurobi(c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray,
                      equality_constraints: Iterable[int] = None,
                      bin_vars: Iterable[int] = None, verbose=False, get_duals=True) -> Optional[
    SolverOutput]:
    r"""
    This is the breakout for solving mixed integer linear programs with gruobi, This is feed directly into the
    MIQP Solver that is defined in the same file.

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

    :return:  A SolverOutput Object
    """
    if not gurobi_pretest(A, b):
        return None

    return solve_miqp_gurobi(c=c, A=A, b=b, equality_constraints=equality_constraints, bin_vars=bin_vars,
                             verbose=verbose, get_duals=get_duals)


def gurobi_pretest(A, b) -> bool:
    """
    Simple shortcuts that indicate an unbounded or infeasible LP

    :param A: LHS Matrix
    :param b: RHS Vector
    :return: True is not trivial unbounded or infeasible constraints
    """
    if A is None or b is None:
        return False

    if numpy.size(A) == 0 or numpy.size(b) == 0:
        return False

    return True
