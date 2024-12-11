from typing import Optional, Sequence, List

import numpy

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from ..solver_interface.solver_interface_utils import (
    SolverOutput,
    get_program_parameters,
)

Matrix = Optional[numpy.ndarray]


def get_bounds_gurobi(A: Matrix, b: Matrix):
    num_x = A.shape[1]
    lb = -GRB.INFINITY * numpy.ones((num_x, 1))
    ub = GRB.INFINITY * numpy.ones((num_x, 1))

    bound_rows = [(i, row) for (i, row) in enumerate(A) if numpy.count_nonzero(row) == 1]
    for row in bound_rows:
        constraint_idx = row[0]
        variable_idx = numpy.nonzero(row[1])[0][0]
        bound = b.flatten()[constraint_idx]/row[1][variable_idx]
        # upper bound
        if row[1][variable_idx] > 0:
            if bound < ub[variable_idx]:
                ub[variable_idx] = bound
        # lower bound
        else:
            if bound > lb[variable_idx]:
                lb[variable_idx] = bound

    return lb, ub


def solve_miqcqp_gurobi(Q: Matrix = None, c: Matrix = None, A: Matrix = None,
                      b: Matrix = None,
                      Q_q: List[Matrix] = None, A_q: Matrix = None, b_q: Matrix = None,
                      equality_constraints: Optional[Sequence[int]] = None,
                      q_equality_constraints: Optional[Sequence[int]] = None,
                      bin_vars: Optional[Sequence[int]] = None, verbose: bool = False,
                      get_duals: bool = True) -> Optional[SolverOutput]:
    r"""
    This is the breakout for solving mixed integer quadratically constrained quadratic programs with gruobi

    The Mixed Integer Quadratically Constrained Quadratic program programming problem

    .. math::

        \min_{xy} \frac{1}{2} [xy]^TQ[xy] + c^T[xy]

    .. math::
        \begin{align}
        A[xy] &\leq b\\
        A_{eq}[xy] &= b_{eq}\\
        [xy]^TQ_q[xy] + A_q[xy] &\leq b_q \ forall q\\
        x &\in R^n\\
        y &\in \{0, 1\}^m
        \end{align}

    :param Q: Square matrix, can be None
    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param Q_q: List of quadratic constraint matrices
    :param A_q: Constraint LHS matrix for quadratic constraints
    :param b_q: Constraint RHS matrix for quadratic constraints
    :param equality_constraints: List of Equality constraints
    :param q_equality_constraints: List of Equality constraints for quadratic constraints
    :param bin_vars: List of binary variable indices
    :param verbose: Flag for output of underlying Solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

    :return: A solver object relating to the solution of the optimization problem
    """
    model = gp.Model()

    if not verbose:
        model.setParam("OutputFlag", 0)

    if equality_constraints is None:
        equality_constraints = []

    if q_equality_constraints is None:
        q_equality_constraints = []

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

    # in the case of non-convex quadratic constraints add the non-convex flag, set the MIP gap to a small tolerance as QCQPs are harder to solve
    if Q_q is not None:
        if get_duals:
            model.Params.QCPDual = 1
        if numpy.min([numpy.min(numpy.linalg.eigvalsh(Q)) for Q in Q_q]) < 0 or len(q_equality_constraints) > 0:
            model.Params.NonConvex = 2
            # noinspection SpellCheckingInspection
            model.Params.MIPgap = 6e-8

    # define num variables and num constraints variables
    num_vars, num_linear_constraints, num_quadratic_constraints = get_program_parameters(Q, c, A, b, Q_q, A_q, b_q)

    if A is None and Q is None and Q_q is None:
        return None

    var_types = [GRB.BINARY if i in bin_vars else GRB.CONTINUOUS for i in range(num_vars)]
    # noinspection PyTypeChecker
    lb = -GRB.INFINITY * numpy.ones((num_vars, 1))
    ub = GRB.INFINITY * numpy.ones((num_vars, 1))
    # if we have quadratic constraints, providing variable bounds might be helpful to gurobi, try to infer them from linear constraints
    if A is not None and num_quadratic_constraints > 0:
        lb, ub = get_bounds_gurobi(A, b)
    # x = model.addMVar(num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=var_types)
    x = model.addMVar(num_vars, lb=lb.flatten(), ub=ub.flatten(), vtype=var_types)

    if num_linear_constraints != 0:
        # sense = numpy.chararray(num_constraints)
        sense = [GRB.LESS_EQUAL for _ in range(num_linear_constraints)]
        for i in equality_constraints:
            sense[i] = GRB.EQUAL

        # sense.fill(GRB.LESS_EQUAL)
        # sense[equality_constraints] = GRB.EQUAL
        # inequality = [i for i in range(num_constraints) if i not in equality_constraints]
        # sense[inequality] = GRB.LESS_EQUAL

        model.addMConstr(A, x, sense, b.flatten())

    if num_quadratic_constraints != 0:
        sense = [GRB.LESS_EQUAL for _ in range(num_quadratic_constraints)]
        for i in q_equality_constraints:
            sense[i] = GRB.EQUAL
        if A_q is None:
            A_q = numpy.zeros((num_quadratic_constraints, num_vars))
        for i in range(num_quadratic_constraints):
            model.addMQConstr(Q_q[i], A_q[i,:].flatten(), sense[i], b_q[i], x, x, x)

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
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None

    # create the Solver return object
    sol = SolverOutput(obj=model.getAttr("ObjVal"), sol=numpy.array(x.X), slack=None,
                       active_set=None, dual=None)

    # if we have a constrained system we need to add in the slack variables and active set
    if num_linear_constraints != 0:

        if get_duals:
            # dual variables only really make sense if the system doesn't have binaries
            # TODO we should also try to check if the problem is convex, since Gurobi doesn't return duals for non-convex problems
            # TODO we should also query duals for the quadratic constraints
            if len(bin_vars) == 0:
                sol.dual = numpy.array(model.getAttr("Pi"))

        # TODO slack and active set also needed for quadratic constraints
        sol.slack = numpy.array(model.getAttr("Slack"))
        sol.active_set = numpy.where((A @ sol.sol.flatten() - b.flatten()) ** 2 < 10 ** -12)[0]

    return sol

def solve_miqp_gurobi(Q: Matrix = None, c: Matrix = None, A: Matrix = None,
                      b: Matrix = None,
                      equality_constraints: Optional[Sequence[int]] = None,
                      bin_vars: Optional[Sequence[int]] = None, verbose: bool = False,
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

    :return: A solver object relating to the solution of the optimization problem
    """
    return solve_miqcqp_gurobi(Q=Q, c=c, A=A, b=b, equality_constraints=equality_constraints, bin_vars=bin_vars, verbose=verbose, get_duals=get_duals)

def solve_qp_gurobi(Q: Matrix, c: Matrix, A: Matrix, b: Matrix,
                    equality_constraints: Optional[Sequence[int]] = None,
                    verbose:bool = False,
                    get_duals:bool = True) -> Optional[SolverOutput]:
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
    return solve_miqcqp_gurobi(Q=Q, c=c, A=A, b=b, equality_constraints=equality_constraints, verbose=verbose,
                             get_duals=get_duals)


# noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
def solve_lp_gurobi(c: Matrix, A: Matrix, b: Matrix,
                    equality_constraints: Optional[Sequence[int]] = None,
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

    return solve_miqcqp_gurobi(c=c, A=A, b=b, equality_constraints=equality_constraints, verbose=verbose,
                             get_duals=get_duals)


def solve_milp_gurobi(c: Matrix, A: Matrix, b: Matrix,equality_constraints: Optional[Sequence[int]] = None,
                      bin_vars: Optional[Sequence[int]] = None, verbose=False, get_duals=True) -> Optional[SolverOutput]:
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

    return solve_miqcqp_gurobi(c=c, A=A, b=b, equality_constraints=equality_constraints, bin_vars=bin_vars,
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
