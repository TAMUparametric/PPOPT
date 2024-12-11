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
class ImplicitCriticalRegion:
    r"""
    Implicit critical region implicitly defines a region in the uncertainty space
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

    # TODO update the docstring above

    # TODO for now, we use sympy here. Since we know structure here, we should avoid it in the future for performance.

    grad_lagrangian: sympy.Matrix
    active_linear_constraints: sympy.Matrix
    active_quadratic_constraints: sympy.Matrix
    inactive_linear_constraints: sympy.Matrix
    inactive_quadratic_constraints: sympy.Matrix
    theta_bounds: sympy.Matrix

    active_set: List[int]

    y_fixation: Optional[numpy.ndarray] = None
    y_indices: Optional[numpy.ndarray] = None
    x_indices: Optional[numpy.ndarray] = None


    # TODO update
    def __repr__(self):
        """Returns a String representation of a Critical Region."""

        # create the output string

        # TODO update this

        output = f"Implicit critical region with active set {self.active_set}"
        output += f"\nThe gradient of the Lagrangian is {self.grad_lagrangian}"
        output += f"\nThe active linear constraints are {self.active_linear_constraints}"
        output += f"\nThe active quadratic constraints are {self.active_quadratic_constraints}"
        output += f"\nThe inactive linear constraints are {self.inactive_linear_constraints}"
        output += f"\nThe inactive quadratic constraints are {self.inactive_quadratic_constraints}"
        output += f"\nThe theta bounds are {self.theta_bounds}"

        return output

    # # TODO update
    # def evaluate(self, theta: numpy.ndarray) -> numpy.ndarray:
    #     """Evaluates x(θ)."""

    #     # if there are not any binary variables in this problem evaluate and return
    #     if self.y_fixation is None:
    #         return numpy.array(self.x_star_numpy(theta)).reshape(-1, 1)

    #     # otherwise evalute x for the continuous variables, then slice in the binaries at the correct locations
    #     cont_vars = numpy.array(self.x_star_numpy(theta))

    #     x_star = numpy.zeros((len(self.x_indices) + len(self.y_indices),))
    #     x_star[self.x_indices] = cont_vars.flatten()
    #     x_star[self.y_indices] = self.y_fixation
    #     return x_star.reshape(-1, 1)

    # # TODO update
    # def lagrange_multipliers(self, theta: numpy.ndarray) -> numpy.ndarray:
    #     """Evaluates λ(θ)."""
    #     return numpy.array(self.lambda_star_numpy(theta))

    # TODO update
    # def is_inside(self, theta: numpy.ndarray, tol: float = 1e-5) -> bool:
    #     """Tests if point θ is inside the critical region."""
    #     # check if all constraints are satisfied
    #     return numpy.all(numpy.array(self.theta_constraints_numpy(theta)) < tol)

    # TODO refactor this to have gurobi stuff outside of here
    # def is_full_dimension(self) -> bool:
    #     """Tests dimensionality of critical region. This is done by checking if the slack of all constraints is positive.

    #     :return: a boolean value, of whether the critical region is full dimensional
    #     """

    #     # should not really happen but if there is an equality constraint, then the region is not full dimensional
    #     for c in self.theta_constraints:
    #         if isinstance(c, sympy.Equality):
    #             return False

    #     slacks = sympy.symbols(f'slack:{len(self.theta_constraints)}')
    #     min_slack = sympy.symbols('min_slack')
        
    #     constraints_with_slack = []
    #     for i, c in enumerate(self.theta_constraints):
    #         constraints_with_slack.append(c.lhs - c.rhs + slacks[i] <= 0)
    #         constraints_with_slack.append(slacks[i] >= 0)
    #         constraints_with_slack.append(slacks[i] >= min_slack)

    #     constraint_strings = [str(c) for c in constraints_with_slack]
    #     syms = []
    #     for c in constraints_with_slack:
    #         syms.extend(c.free_symbols)
    #     syms = list(set(syms))
    #     syms.sort(key=str)

    #     # TODO this is ugly and should be done nicer later
    #     for i_con, c in enumerate(constraint_strings):
    #         constraint_strings[i_con] = c.replace('<=', '==')

    #     replacement_dict, constraint_strings, num_aux = replace_square_roots_dictionary(constraint_strings)

    #     model = build_gurobi_model_with_square_roots(constraint_strings, syms, replacement_dict, num_aux)

    #     min_slack_var = model.getVarByName('min_slack')
    #     model.setObjective(min_slack_var, gurobipy.GRB.MAXIMIZE)

    #     model.optimize()
    #     status = model.status
    #     if status != gurobipy.GRB.OPTIMAL:
    #         return False
        
    #     return model.objVal > 1e-8

    # # TODO update
    # def get_constraints(self):
    #     """
    #     An assessor function to quickly access the fields of the extends of the critical region

    #     :return: the constraints on theta as symbolic expressions
    #     """
    #     return self.theta_constraints
