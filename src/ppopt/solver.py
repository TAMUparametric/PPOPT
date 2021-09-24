
import importlib.util
import numpy
import sys

from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable

from .solver_interface.cvxopt_interface import solve_lp_cvxopt
from .solver_interface.gurobi_solver_interface import solve_lp_gurobi, solve_qp_gurobi, solve_milp_gurobi, \
    solve_miqp_gurobi
from .solver_interface.quad_prog_interface import solve_qp_quadprog
from .solver_interface.solver_interface_utils import SolverOutput

def check_modules(modules: Iterable):
    return [module for module in modules if module in sys.modules]


def check_solver_modules(module_map, packages):
    avalable_packages = check_modules(packages)
    return [module_map[package] for package in avalable_packages]


def avalable_LP_solvers():
    solver_map = {'cvxopt': 'glpk', 'gurobipy': 'gurobi'}
    check_packages = ['cvxopt', 'gurobipy']
    return check_solver_modules(solver_map, check_packages)


def avalable_QP_solvers():
    solver_map = {'quadprog': 'quadprog', 'gurobipy': 'gurobi'}
    check_packages = ['quadprog', 'gurobipy']
    return check_solver_modules(solver_map, check_packages)

def default_solver_options():
    default_solver = {'lp': 'gurobi', 'qp': 'gurobi', 'milp': 'gurobi', 'miqp': 'gurobi'}

    if 'glpk' in avalable_LP_solvers():
        default_solver['lp'] = 'glpk'

    if 'quadprog' in avalable_LP_solvers():
        default_solver['qp'] = 'quadprog'

    return default_solver


@dataclass
class Solver:
    """
    This is the primary user interface for deterministic solvers
    """
    solvers: Dict[str, str] = field(default_factory=default_solver_options)

    supported_problems = ['lp', 'qp', 'milp', 'miqp']
    supported_solvers = ['gurobi', 'glpk', 'quadprog']

    def __post_init__(self):
        """
        If the user gives only specifies some solvers then we need to make sure that whe can handle that instance or \\
        handle the case that the Solver is not supported
        """

        # check that the
        for pair in self.solvers.items():

            if pair[0] not in self.supported_problems:
                self.problem_not_supported(pair[0])

            if pair[1] not in self.supported_solvers:
                self.solver_not_supported(pair[1])

    @staticmethod
    def problem_not_supported(problem_name: str) -> None:
        """This is an internal method that throws an error and prompts the user when they use an unsupported Solver"""

        message = f"Problem {problem_name} is not supported! \n" \
                  + f'MPO Supports the following problems {str(Solver.supported_problems)} \n' \
                  + f'If you have a misspelled a supported problem, please make sure you spelled it correctly \n'

        raise RuntimeError(message)

    @staticmethod
    def solver_not_supported(solver_name: str) -> None:
        """This is an internal method that throws an error and prompts the user when they use an unsupported Solver"""

        message = f"Solver {solver_name} is not supported! \n" \
                  + f'MPO Supports the following solvers {str(Solver.supported_solvers)} \n' \
                  + f'If you have a supported Solver, please change the default ppopt Solver to your specific Solver when you load the package \n' \
                  + f'mpo.settings.optimization_solver = \'solver_name\''

        raise RuntimeError(message)

    def check_supported_problem(self, problem_name: str) -> None:
        if problem_name not in self.solvers:
            message = f"Problem {problem_name} is has not been defined for this solver! \n" \
                      + f'This solver has the following problem types defined {str(self.solvers.items())} \n' \
                      + f'If this is one of the supported problems {str(Solver.supported_problems)} then simply add it when defining the solver object\n'
            raise RuntimeError(message)

    # noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
    def solve_miqp(self, Q: Optional[numpy.ndarray], c: Optional[numpy.ndarray], A: Optional[numpy.ndarray],
                   b: Optional[numpy.ndarray],
                   equality_constraints: Iterable[int] = None,
                   bin_vars: Iterable[int] = None, verbose: bool = False,
                   get_duals: bool = True) -> Optional[SolverOutput]:
        r"""
        This is the breakout for solving mixed integer quadratic programs

        The Mixed Integer Quadratic program programming problem
            min_{xy} 1/2 [xy]^T@Q@[xy] + c^T@[xy]

            s.t.   A@[xy] <= b
                   A_eq@[xy] = beq

                   xy is the parameter vector of mixed real and binary inputs
                   x \in R^n
                   y \in \{0, 1\}^m

        :param Q: Square matrix, can be None
        :param c: Column Vector, can be None
        :param A: Constraint LHS matrix, can be None
        :param b: Constraint RHS matrix, can be None
        :param equality_constraints: List of Equality constraints
        :param bin_vars: List of binary variable indices
        :param verbose: Flag for output of underlying Solver, default False
        :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

        :return: A SolverOutput object if optima found, otherwise None.
        """

        if self.solvers['miqp'] == "gurobi":
            return solve_miqp_gurobi(Q, c, A, b, equality_constraints, bin_vars, verbose, get_duals)

        else:
            self.solver_not_supported(self.solvers['miqp'])

    def solve_qp(self, Q: Optional[numpy.ndarray], c: Optional[numpy.ndarray], A: Optional[numpy.ndarray],
                 b: Optional[numpy.ndarray], equality_constraints: Iterable[int] = None,
                 verbose=False,
                 get_duals=True) -> Optional[SolverOutput]:
        r"""
        This is the breakout for solving quadratic programs

        The Quadratic programming problem
            min_{x} 1/2 x^T@Q@x + c^T@x

            s.t.   A@x <= b
                   A_eq@x = beq

                   x \in R^n

        :param Q: Square matrix, can be None
        :param c: Column Vector, can be None
        :param A: Constraint LHS matrix, can be None
        :param b: Constraint RHS matrix, can be None
        :param equality_constraints: List of Equality constraints
        :param verbose: Flag for output of underlying Solver, default False
        :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

        :return: A SolverOutput object if optima found, otherwise None.
        """

        if self.solvers['qp'] == "gurobi":
            return solve_qp_gurobi(Q, c, A, b, equality_constraints, verbose, get_duals)
        elif self.solvers['qp'] == "quadprog":
            return solve_qp_quadprog(Q, c, A, b, equality_constraints, verbose, get_duals)
        else:
            self.solver_not_supported(self.solvers['qp'])
            return None

    # noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
    def solve_lp(self, c: Optional[numpy.ndarray], A: Optional[numpy.ndarray], b: Optional[numpy.ndarray],
                 equality_constraints=None, verbose=False,
                 get_duals=True) -> Optional[SolverOutput]:
        r"""
        This is the breakout for solving linear programs

        The Linear programming problem
            min_{xy} c^T@[xy]

            s.t.   A@x <= b
                   A_eq@x = beq

                   x \in R^n

        :param c: Column Vector, can be None
        :param A: Constraint LHS matrix, can be None
        :param b: Constraint RHS matrix, can be None
        :param equality_constraints: List of Equality constraints
        :param verbose: Flag for output of underlying Solver, default False
        :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

        :return: A SolverOutput object if optima found, otherwise None.
        """

        if self.solvers['lp'] == "gurobi":
            return solve_lp_gurobi(c, A, b, equality_constraints, verbose, get_duals)
        if self.solvers['lp'] == 'glpk':
            return solve_lp_cvxopt(c, A, b, equality_constraints, verbose, get_duals)
        else:
            self.solver_not_supported(self.solvers['lp'])

    def solve_milp(self, c: Optional[numpy.ndarray], A: Optional[numpy.ndarray], b: Optional[numpy.ndarray],
                   equality_constraints: Iterable[int] = None,
                   bin_vars: Iterable[int] = None, verbose=False, get_duals=True) -> Optional[SolverOutput]:
        r"""
        This is the breakout for solving mixed integer linear programs

        The Mixed Integer Linear programming problem
            min_{xy} c^T*[xy]

            s.t.   A[xy] <= b
                   Aeq*[xy] = beq

                   xy is the parameter vector of mixed real and binary inputs
                   x \in R^n
                   y \in \{0, 1\}^m

        :param c: Column Vector, can be None
        :param A: Constraint LHS matrix, can be None
        :param b: Constraint RHS matrix, can be None
        :param equality_constraints: List of Equality constraints
        :param bin_vars: List of binary variable indices
        :param verbose: Flag for output of underlying Solver, default False
        :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

        :return: A dictionary of the Solver outputs, or none if infeasible or unbounded. output['sol'] = primal variables, output['dual'] = dual variables, output['obj'] = objective value, output['const'] = slacks, output['active'] = active constraints.
        """

        if self.solvers['milp'] == "gurobi":
            return solve_milp_gurobi(c, A, b, equality_constraints, bin_vars, verbose, get_duals)
        else:
            self.solver_not_supported(self.solvers['milp'])
