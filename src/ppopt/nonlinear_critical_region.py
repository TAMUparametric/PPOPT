from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy
import sympy
import gurobipy

from .utils.chebyshev_ball import chebyshev_ball

from .utils.symbolic_utils import (
    replace_square_roots_dictionary,
    build_gurobi_model_with_square_roots,
)

# TODO many methods of this class will not yet work as they need to be updated to reflect that this is a nonlinear and symbolic CR rather than a polytope


@dataclass(eq=False)
class NonlinearCriticalRegion:
    r"""
    Nonlinear critical region is a nonlinear set that defines a region in the uncertainty space
    with an associated optimal value, active set, lagrange multipliers and
    constraints

    .. math::

        \begin{align}
            x(\theta) &= A\theta + b\\
            \lambda(\theta) &= C\theta + d\\
            \Theta &:= \{\forall \theta \in \mathbf{R}^m: E\theta \leq f\}
        \end{align}

    equality_indices: numpy array of indices

    constraint_set: if this is an A@x = b + F@theta boundary

    lambda_set: if this is a λ = 0 boundary

    boundary_set: if this is an Eθ <= f boundary

    """

    # TODO update the docstring above

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

    def __init__(self, x_star, lambda_star, theta_constraints, active_set, y_fixation=None, y_indices=None, x_indices=None):
        self.x_star = x_star
        self.lambda_star = lambda_star
        self.theta_constraints = theta_constraints
        self.active_set = active_set
        self.y_fixation = y_fixation
        self.y_indices = y_indices
        self.x_indices = x_indices

        theta_syms = []
        for c in self.theta_constraints:
            theta_syms.extend(c.free_symbols)
        num_theta = len(list(set(theta_syms))) # get the number of unique theta variables
        theta = sympy.Matrix(sympy.symbols(f'theta:{num_theta}'))
        self.x_star_numpy = sympy.lambdify([theta], self.x_star, 'numpy')
        self.lambda_star_numpy = sympy.lambdify([theta], self.lambda_star, 'numpy')
        self.theta_constraints_numpy = sympy.lambdify([theta], [c.lhs - c.rhs for c in self.theta_constraints], 'numpy')

    def __repr__(self):
        """Returns a String representation of a Critical Region."""

        # create the output string

        # TODO update this

        output = f"Critical region with active set {self.active_set}"
        output += f"\nThe Omega Constraint indices are {self.omega_set}"
        output += f"\nThe Lagrange multipliers Constraint indices are {self.lambda_set}"
        output += f"\nThe Regular Constraint indices are {self.regular_set}"
        output += "\n  x(θ) = Aθ + b \n λ(θ) = Cθ + d \n  Eθ <= f"
        output += f"\n A = {self.A} \n b = {self.b} \n C = {self.C} \n d = {self.d} \n E = {self.E} \n f = {self.f}"

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

    # depreciated
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
