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
        """Returns a String output of Critical Region."""
        return f"Critical region with active set {self.active_set}\nThe Omega Constraint indices are {self.omega_set}\nThe Lagrange multipliers Constraint indices are {self.lambda_set}\nThe Regular Constraint indices are {self.regular_set}\n  x(θ) = Aθ + b \n λ(θ) = Cθ + d \n  Eθ <= f \n A = {self.A} \n b = {self.b} \n C = {self.C} \n d = {self.d} \n E = {self.E} \n f = {self.f}"

    def evaluate(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates x(θ) = Aθ + b."""

        if self.y_fixation is not None:
            cont_vars = self.A @ theta + self.b
            x_star = numpy.zeros((len(self.x_indices) + len(self.y_indices),))
            x_star[self.x_indices] = cont_vars.flatten()
            x_star[self.y_indices] = self.y_fixation
            return x_star.reshape(-1, 1)
        else:
            return self.A @ theta + self.b

    def lagrange_multipliers(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates λ(θ) = Cθ + d."""
        return self.C @ theta + self.d

    def is_inside(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Tests if point θ is inside the critical region."""
        return numpy.all(self.E @ theta - self.f < 0)

    # depreciated
    def is_full_dimension(self) -> bool:
        """Tests dimensionality of critical region."""
        # I think so

        soln = chebyshev_ball(self.E, self.f)
        if soln is not None:
            return soln.sol[-1] > 10 ** -8
        return False

    def get_constraints(self):
        return [self.E, self.f]
