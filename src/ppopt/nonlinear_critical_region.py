from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy
import sympy
import gurobipy

from .utils.chebyshev_ball import chebyshev_ball

from .utils.symbolic_utils import (
    replace_square_roots_dictionary,
    build_gurobi_model_with_square_roots,
    to_less_than_or_equal,
)


@dataclass(eq=False)
class NonlinearCriticalRegion:
    r"""
    Nonlinear critical region is a nonlinear set that defines a region in the uncertainty space
    with an associated optimal value, active set, lagrange multipliers and
    constraints

    .. math::

        \begin{align}
            x^* &= x(\theta)\\
            \lambda^* &= \lambda(\theta)\\
            \Theta &:= \{\forall \theta \in \mathbf{R}^m: h(\theta) \leq 0\}
        \end{align}

    active_set: numpy array of indices

    regular_set: if this is an x(theta) = 0 boundary

    lambda_set: if this is a λ = 0 boundary

    omega_set: if this is an h(theta) = 0 boundary

    """

    x_star: sympy.Matrix
    lambda_star: sympy.Matrix
    theta_constraints: List[sympy.core.relational.LessThan]

    active_set: List[int]

    x_star_numpy: Optional[Callable] = None
    lambda_star_numpy: Optional[Callable] = None
    theta_constraints_numpy: Optional[Callable] = None

    omega_set: List[int] = field(default_factory=list)
    lambda_set: List[int] = field(default_factory=list)
    regular_set: List[List[int]] = field(default_factory=list)

    y_fixation: Optional[numpy.ndarray] = None
    y_indices: Optional[numpy.ndarray] = None
    x_indices: Optional[numpy.ndarray] = None

    def __init__(self, x_star, lambda_star, theta_constraints, active_set, omega_set, lambda_set, regular_set, y_fixation=None, y_indices=None, x_indices=None):
        self.x_star = x_star
        self.lambda_star = lambda_star
        self.theta_constraints = [to_less_than_or_equal(c) for c in theta_constraints] # having all constraints as <= is helpful for building the numeric evaluation, since we can then always do lhs-rhs
        self.active_set = active_set
        self.omega_set = omega_set
        self.lambda_set = lambda_set
        self.regular_set = regular_set
        self.y_fixation = y_fixation
        self.y_indices = y_indices
        self.x_indices = x_indices

        theta_syms = []
        for c in self.theta_constraints:
            theta_syms.extend(c.free_symbols)
        num_theta = len(list(set(theta_syms))) # get the number of unique theta variables
        beta = sympy.symbols('beta')
        if 'beta' in [str(t) for t in theta_syms]:
            num_theta -= 1
            theta_syms.sort(key=str)
            beta = theta_syms[0]
        theta = sympy.Matrix(sympy.symbols(f'theta:{num_theta}'))
        x_star_beta = [x.subs({beta:1}) for x in self.x_star]
        lambda_star_beta = [l.subs({beta:1}) for l in self.lambda_star]
        theta_constraints_beta = [c.subs({beta:1}) for c in self.theta_constraints]
        self.x_star_numpy = sympy.lambdify([theta], x_star_beta, 'numpy')
        self.lambda_star_numpy = sympy.lambdify([theta], lambda_star_beta, 'numpy')
        self.theta_constraints_numpy = sympy.lambdify([theta], [c.lhs - c.rhs for c in theta_constraints_beta], 'numpy')

    def __repr__(self):
        """Returns a String representation of a Critical Region."""

        # create the output string

        # TODO update this

        output = f"Critical region with active set {self.active_set}"
        output += f"\nThe Omega Constraint indices are {self.omega_set}"
        output += f"\nThe Lagrange multipliers Constraint indices are {self.lambda_set}"
        output += f"\nThe Regular Constraint indices are {self.regular_set}"
        output += f"\nx(θ) = \n{numpy.array(self.x_star).reshape(-1, 1)}"
        output += f"\nλ(θ) = \n{numpy.array(self.lambda_star).reshape(-1, 1)}"
        output += f"\nConstraints on θ are {self.theta_constraints}"

        return output

    def evaluate(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates x(θ)."""

        # if there are not any binary variables in this problem evaluate and return
        if self.y_fixation is None:
            return numpy.array(self.x_star_numpy(theta)).reshape(-1, 1)

        # otherwise evalute x for the continuous variables, then slice in the binaries at the correct locations
        cont_vars = numpy.array(self.x_star_numpy(theta))

        x_star = numpy.zeros((len(self.x_indices) + len(self.y_indices),))
        x_star[self.x_indices] = cont_vars.flatten()
        x_star[self.y_indices] = self.y_fixation
        return x_star.reshape(-1, 1)

    def lagrange_multipliers(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates λ(θ)."""
        return numpy.array(self.lambda_star_numpy(theta))

    def is_inside(self, theta: numpy.ndarray, tol: float = 1e-5) -> bool:
        """Tests if point θ is inside the critical region."""
        # check if all constraints are satisfied
        return numpy.all(numpy.array(self.theta_constraints_numpy(theta)) < tol)

    # TODO refactor this to have gurobi stuff outside of here
    def is_full_dimension(self) -> bool:
        """Tests dimensionality of critical region. This is done by checking if the slack of all constraints is positive.

        :return: a boolean value, of whether the critical region is full dimensional
        """

        # should not really happen but if there is an equality constraint, then the region is not full dimensional
        for c in self.theta_constraints:
            if isinstance(c, sympy.Equality):
                return False

        slacks = sympy.symbols(f'slack:{len(self.theta_constraints)}')
        min_slack = sympy.symbols('min_slack')
        
        constraints_with_slack = []
        for i, c in enumerate(self.theta_constraints):
            constraints_with_slack.append(c.lhs - c.rhs + slacks[i] <= 0)
            constraints_with_slack.append(slacks[i] >= 0)
            constraints_with_slack.append(slacks[i] >= min_slack)

        constraint_strings = [str(c) for c in constraints_with_slack]
        syms = []
        for c in constraints_with_slack:
            syms.extend(c.free_symbols)
        syms = list(set(syms))
        syms.sort(key=str)

        # TODO this is ugly and should be done nicer later
        for i_con, c in enumerate(constraint_strings):
            constraint_strings[i_con] = c.replace('<=', '==')

        replacement_dict, constraint_strings, num_aux = replace_square_roots_dictionary(constraint_strings)

        model = build_gurobi_model_with_square_roots(constraint_strings, syms, replacement_dict, num_aux)

        min_slack_var = model.getVarByName('min_slack')
        model.setObjective(min_slack_var, gurobipy.GRB.MAXIMIZE)

        model.optimize()
        status = model.status
        if status != gurobipy.GRB.OPTIMAL:
            return False
        
        return model.objVal > 1e-8

    def get_constraints(self):
        """
        An assessor function to quickly access the fields of the extends of the critical region

        :return: the constraints on theta as symbolic expressions
        """
        return self.theta_constraints
