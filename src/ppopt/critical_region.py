from dataclasses import dataclass, field
from typing import List, Union

import numpy

from .utils.chebyshev_ball import chebyshev_ball


@dataclass
class CriticalRegion:
    r"""
    Critical region is a polytope that defines a region in the uncertainty space
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

    A: numpy.ndarray
    b: numpy.ndarray
    C: numpy.ndarray
    d: numpy.ndarray
    E: numpy.ndarray
    f: numpy.ndarray
    active_set: Union[List[int], numpy.ndarray]

    omega_set: Union[List[int], numpy.ndarray] = field(default_factory=list)
    lambda_set: Union[List[int], numpy.ndarray] = field(default_factory=list)
    regular_set: Union[List[int], numpy.ndarray] = field(default_factory=list)

    y_fixation: numpy.ndarray = None
    y_indices: numpy.ndarray = None
    x_indices: numpy.ndarray = None

    def __repr__(self):
        """Returns a String representation of a Critical Region."""

        # create the output string

        output = f"Critical region with active set {self.active_set}"
        output += f"\nThe Omega Constraint indices are {self.omega_set}"
        output += f"\nThe Lagrange multipliers Constraint indices are {self.lambda_set}"
        output += f"\nThe Regular Constraint indices are {self.regular_set}"
        output += "\n  x(θ) = Aθ + b \n λ(θ) = Cθ + d \n  Eθ <= f"
        output += f"\n A = {self.A} \n b = {self.b} \n C = {self.C} \n d = {self.d} \n E = {self.E} \n f = {self.f}"

        return output

    def evaluate(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates x(θ) = Aθ + b."""

        # if there are not any binary variables in this problem evaluate and return
        if self.y_fixation is None:
            return self.A @ theta + self.b

        # otherwise evalute AΘ+b for the continuous variables, then slice in the binaries at the correct locations
        cont_vars = self.A @ theta + self.b
        x_star = numpy.zeros((len(self.x_indices) + len(self.y_indices),))
        x_star[self.x_indices] = cont_vars.flatten()
        x_star[self.y_indices] = self.y_fixation
        return x_star.reshape(-1, 1)

    def lagrange_multipliers(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates λ(θ) = Cθ + d."""
        return self.C @ theta + self.d

    def is_inside(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Tests if point θ is inside the critical region."""
        # check if all constraints EΘ <= f
        return numpy.all(self.E @ theta - self.f < 0)

    # depreciated
    def is_full_dimension(self) -> bool:
        """Tests dimensionality of critical region. This is done by checking the radius of the chebyshev ball inside
        the region

        :return: a boolean value, of whether the critical region is full dimensional
        """

        # solve the chebyshev ball LP
        soln = chebyshev_ball(self.E, self.f)

        # if this is infeasible, then it definitely is not full dimension as it is empty and doesn't have a good
        # dimensional description
        if soln is None:
            return False

        # if the chebyshev LP is feasible then we check if the radius is larger than some epsilon value
        return soln.sol[-1] > 10 ** -8

    def get_constraints(self):
        """
        An assessor function to quickly access the fields of the extends of the critical region

        :return: a list with E, and f as elements
        """
        return [self.E, self.f]
