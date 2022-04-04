from typing import List, Optional

import numpy

from .mpmilp_program import MPMILP_Program
from .mpqp_program import MPQP_Program
from .solver import Solver


class MPMIQP_Program(MPMILP_Program):
    r"""
    The standard class for  multiparametric mixed integer quadratic programming.

    .. math::
        \min \frac{1}{2}x^T Qx + \theta^T H^T x + c^T x + c_c + c_t^T \theta + \frac{1}{2} \theta^T Q_t\theta

    .. math::
        \begin{align}
        A_{eq}x &= b_{eq} + F_{eq}\theta\\
        Ax &\leq b + F\theta\\
        A_\theta \theta &\leq b_\theta\\
        x_i &\in \mathbb{R} \text{ or } \mathbb{B}\\
        \end{align}

    Equality constraints containing only binary variables cannot also be parametric, as that generate a non-convex and
    discrete feasible parameter space

    """

    def __init__(self, A: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray, H: numpy.ndarray, Q: numpy.ndarray,
                 A_t: numpy.ndarray,
                 b_t: numpy.ndarray, F: numpy.ndarray, binary_indices: List, c_c: Optional[numpy.ndarray] = None,
                 c_t: Optional[numpy.ndarray] = None, Q_t: Optional[numpy.ndarray] = None,
                 equality_indices=None, solver: Solver = Solver()):
        """Initialized the MPMIQP_Program."""
        # calls MPMILP_Program's constructor to reduce out burden
        super(MPMIQP_Program, self).__init__(A, b, c, H, A_t, b_t, F, binary_indices, c_c, c_t, Q_t, equality_indices,
                                             solver)
        self.Q = Q
        self.constraint_datatype_conversion()

    def evaluate_objective(self, x: numpy.ndarray, theta_point: numpy.ndarray):
        """Evaluates the objective f(x,theta)"""
        return 0.5 * x.T @ self.Q @ x + theta_point.T @ self.H.T @ x + self.c.T @ x + self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point

    def generate_substituted_problem(self, fixed_combination: List[int]):
        """
        Generates the fixed binary continuous version of the problem e.g. substitute all the binary variables
        :param fixed_combination:
        :return:
        """

        # handle only the constraint matrices for now
        A_cont = self.A[:, self.cont_indices]
        A_bin = self.A[:, self.binary_indices]

        fixed_combination = numpy.array(fixed_combination).reshape(-1, 1)

        # find any integer only constraints and remove them (these are safe to remove as they are not used in the
        # substituted problem)
        kept_constraints = []
        for i in range(self.num_constraints()):

            # constraint of the type sum(y_i, i in I) ?? b -> we do not need this
            if numpy.allclose(A_cont[i], 0 * A_cont[i]) and numpy.allclose(self.F[i], 0 * self.F[i]):
                continue
            kept_constraints.append(i)

        # remove integer only constraints from equality set
        equality_set = [i for i in self.equality_indices if i in kept_constraints]

        A_cont = A_cont[kept_constraints]
        A_bin = A_bin[kept_constraints]
        b = self.b[kept_constraints] - A_bin @ fixed_combination
        F = self.F[kept_constraints]

        Q_c = self.Q[:, self.cont_indices][self.cont_indices]
        Q_d = self.Q[:, self.binary_indices][self.binary_indices]

        H_alpha = self.Q[:, self.cont_indices][self.binary_indices]
        c = self.c[self.cont_indices] + fixed_combination.T @ H_alpha
        c_c = self.c_c + self.c[
            self.binary_indices].T @ fixed_combination + 0.5 * fixed_combination.T @ Q_d @ fixed_combination
        H_c = self.H[self.cont_indices]
        H_d = self.H[self.binary_indices]

        c_t = self.c_t + fixed_combination.T @ H_d

        sub_problem = MPQP_Program(A_cont, b, c, H_c, Q_c, self.A_t, self.b_t, F, c_c, c_t, self.Q_t, equality_set,
                                   self.solver)
        sub_problem.process_constraints(True)
        return sub_problem
