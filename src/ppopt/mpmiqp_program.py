from typing import List, Optional, Union

import numpy

from .mpmilp_program import MPMILP_Program
from .mpqp_program import MPQP_Program
from .solver import Solver
from .solver_interface.solver_interface_utils import SolverOutput


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
                 equality_indices: Optional[List[int]] = None, solver: Optional[Solver] = None,
                 post_process: bool = True):
        """Initialized the MPMIQP_Program."""
        # calls MPMILP_Program's constructor to reduce out burden

        if solver is None:
            solver = Solver()

        self.Q = Q
        super(MPMIQP_Program, self).__init__(A, b, c, H, A_t, b_t, F, binary_indices, c_c, c_t, Q_t, equality_indices,
                                             solver, post_process=False)

        if post_process:
            self.post_process()

    def evaluate_objective(self, x: numpy.ndarray, theta_point: numpy.ndarray) -> float:
        """Evaluates the objective f(x,theta)"""
        obj_val = 0.5 * x.T @ self.Q @ x + theta_point.T @ self.H.T @ x + self.c.T @ x + self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point
        return float(obj_val[0, 0])

    def solve_theta(self, theta_point: numpy.ndarray) -> Optional[SolverOutput]:
        """
        Solves the substituted problem,with the provided theta

        :param theta_point:
        :param deterministic_solver:
        :return:
        """
        soln = self.solver.solve_miqp(self.Q, self.c + self.H @ theta_point, self.A, self.b + self.F @ theta_point,
                                      self.equality_indices, self.binary_indices)
        if soln is not None:
            const_term = self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point
            soln.obj += float(const_term[0, 0])

        return soln

    def generate_substituted_problem(self, fixed_combination: Union[numpy.ndarray, List[int]]):
        """
        Generates the fixed binary continuous version of the problem e.g. substitute all the binary variables
        :param fixed_combination:
        :return:
        """

        # handle only the constraint matrices for now
        A_cont = self.A[:, self.cont_indices]
        A_bin = self.A[:, self.binary_indices]

        fixed_combination = numpy.array(fixed_combination).reshape(-1, 1)

        # helper function to classify constraint types
        def is_not_binary_constraint(i: int):
            return not (numpy.allclose(A_cont[i], 0 * A_cont[i]) and numpy.allclose(self.F[i], 0 * self.F[i]))

        inequality_indices = [i for i in range(self.num_constraints()) if i not in self.equality_indices]

        kept_equality_constraints = list(filter(is_not_binary_constraint, self.equality_indices))
        kept_ineq_constraints = list(filter(is_not_binary_constraint, inequality_indices))

        kept_constraints = [*kept_equality_constraints, *kept_ineq_constraints]

        new_equality_set = [i for i in range(len(kept_equality_constraints))]

        A_cont = A_cont[kept_constraints]
        A_bin = A_bin[kept_constraints]
        b = self.b[kept_constraints] - A_bin @ fixed_combination
        F = self.F[kept_constraints]

        Q_c = self.Q[:, self.cont_indices][self.cont_indices]
        Q_d = self.Q[:, self.binary_indices][self.binary_indices]

        H_alpha = self.Q[:, self.cont_indices][self.binary_indices]
        c = self.c[self.cont_indices] + (H_alpha.T @ fixed_combination)
        c_c = self.c_c + self.c[
            self.binary_indices].T @ fixed_combination + 0.5 * fixed_combination.T @ Q_d @ fixed_combination
        H_c = self.H[self.cont_indices]
        H_d = self.H[self.binary_indices]

        c_t = self.c_t + (fixed_combination.T @ H_d).T

        sub_problem = MPQP_Program(A_cont, b, c, H_c, Q_c, self.A_t, self.b_t, F, c_c, c_t, self.Q_t, new_equality_set,
                                   self.solver)
        return sub_problem
