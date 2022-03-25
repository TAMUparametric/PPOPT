from typing import List, Optional, Tuple

import numpy

from .mplp_program import MPLP_Program
from .solver_interface.solver_interface import SolverOutput
from .utils.general_utils import latex_matrix, remove_size_zero_matrices, ppopt_block, make_column


class MPQP_Program(MPLP_Program):
    r"""
    The standard class for quadratic multiparametric programming.

    .. math::
        \min \frac{1}{2}x^TQx + \theta^TH^Tx + c^Tx
    .. math::
        \begin{align}
        Ax &\leq b + F\theta\\
        A_{eq}x &= b_{eq}\\
        A_\theta \theta &\leq b_\theta\\
        x &\in R^n\\
        \end{align}
    """

    def __init__(self, A: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray, H: numpy.ndarray, Q: numpy.ndarray,
                 A_t: numpy.ndarray,
                 b_t: numpy.ndarray, F: numpy.ndarray, c_c: Optional[numpy.ndarray] = None,
                 c_t: Optional[numpy.ndarray] = None, Q_t: Optional[numpy.ndarray] = None,
                 equality_indices=None, solver=None):
        """Initialized the MPQP_Program."""
        # calls MPLP_Program's constructor to reduce out burden
        super(MPQP_Program, self).__init__(A, b, c, H, A_t, b_t, F, c_c, c_t, Q_t, equality_indices, solver)

        # assignees member variables
        self.Q = Q

        self.constraint_datatype_conversion()
        # calls the MPLP __post_init__ to handle equality constraints
        # super(MPQP_Program, self).__post_init__()

    def evaluate_objective(self, x, theta_point):
        r"""
        Evaluates the objective of the multiparametric program. for a given x and Θ.

        .. math::
            \frac{1}{2}x^TQx + \theta^TH^Tx+c^Tx

        :param x: x input
        :param theta_point: θ input
        :return: Objective function evaluated at x, θ
        """
        return 0.5 * x.T @ self.Q @ x + theta_point.T @ self.H.T @ x + self.c.T @ x + self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point

    def warnings(self) -> List[str]:
        """Checks the dimensions of the matrices to ensure consistency."""
        warning_list = MPLP_Program.warnings(self)

        # Quadratic Problem specific warnings

        # Checks if Q is square
        if self.Q.shape[0] != self.Q.shape[1]:
            warning_list.append(f'Q matrix is not square with dimensions {self.Q.shape}')

        # check if Q matches the number of optimization variables
        if self.Q.shape[0] != self.A.shape[1]:
            warning_list.append('Dimensions of Q and A matrices disagree in number of x parameters')

        if self.Q.shape[1] != self.A.shape[1]:
            warning_list.append('Dimensions of Q and A matrices disagree in number of x parameters')

        # check the condition number of the matrix and make sure it is invertible
        if self.Q.shape[0] == self.Q.shape[1]:
            e_values, _ = numpy.linalg.eig(self.Q)
            if len(e_values) < 0:
                warning_list.append(f'Non-convex quadratic program detected, with eigenvalues {e_values}')
            elif len(e_values) < 10 ** -4:
                warning_list.append(f'Possible positive semi-definite nature detected in Q, eigenvalues {e_values}')

        # return warnings
        return warning_list

    def latex(self) -> List[str]:
        """Creates a latex output for the multiparametric problem."""
        # calls the latex function inherited from MPLP to get most of the output
        output = super(MPQP_Program, self).latex()

        # creates a latex output for a column vector of x variables
        x = [rf'x_{i}' for i in range(self.num_x())]
        theta = [f'\\theta_{i}' for i in range(self.num_t())]

        theta_latex = latex_matrix(theta)
        x_latex = latex_matrix(x)

        added_term = ''
        if not numpy.allclose(self.H, numpy.zeros_like(self.H)):
            added_term = " + " + theta_latex + '^{T}' + latex_matrix(self.H) + x_latex

        # modifies the linear output of the MPLP function to include the quadratic term
        output[0] = "$$" + '\\min_{x} \\frac{1}{2}' + x_latex + '^{T}' + latex_matrix(
            self.Q) + x_latex + '+' + latex_matrix(self.c) + "^T" + x_latex + added_term + "$$"

        return output

    def solve_theta(self, theta_point: numpy.ndarray, deterministic_solver: str = 'gurobi') -> Optional[SolverOutput]:
        r"""
        Substitutes theta into the multiparametric problem and solves the following optimization problem

        .. math::
            \min_{x} \frac{1}{2}x^TQx + \tilde{c}^Tx

        .. math::
            \begin{align}
            Ax &\leq \tilde{b}\\
            A_{eq}x &= \tilde{b}_{eq}\\
            x &\in R^n\\
            \end{align}

        :param theta_point: An uncertainty realization
        :param deterministic_solver: Deterministic solver to use to solve the above quadratic program
        :return: The Solver output of the substituted problem, returns None if not solvable
        """

        if not numpy.all(self.A_t @ theta_point <= self.b_t):
            return None

        sol_obj = self.solver.solve_qp(Q=self.Q, c=self.H @ theta_point + self.c, A=self.A,
                                       b=self.b + (self.F @ theta_point),
                                       equality_constraints=self.equality_indices)

        if sol_obj is not None:
            sol_obj.obj += self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point
            return sol_obj

        return None

    def optimal_control_law(self, active_set: List[int]) -> Tuple:
        r"""
        This function calculates the optimal control law corresponding to an active set combination

        :param active_set: an active set combination
        :return: a tuple of the optimal x* and λ* functions in the following form(A_x, b_x, A_l, b_l)

        .. math::

            \begin{align*}
            x^*(\theta) &= A_x\theta + b_x\\
            \lambda^*(\theta) &= A_l\theta + b_l\\
            \end{align*}
        """

        inverse_Q = numpy.linalg.pinv(self.Q)
        aux = self.A[active_set] @ inverse_Q
        auxinv = numpy.linalg.pinv(aux @ self.A[active_set].T)

        lagrange_A = -auxinv @ (aux @ self.H + self.F[active_set])
        lagrange_b = -auxinv @ (self.b[active_set] + aux @ self.c)

        aux = inverse_Q @ self.A[active_set].T

        parameter_A = - aux @ lagrange_A - inverse_Q @ self.H
        parameter_b = -aux @ lagrange_b - inverse_Q @ self.c

        return parameter_A, parameter_b, lagrange_A, lagrange_b

    # def process_constraints(self, find_implicit_equalities = False) -> None:
    #     super(MPQP_Program, self).process_constraints(find_implicit_equalities = False)

    def check_optimality(self, active_set: list):
        r"""
        Tests if the active set is optimal for the provided mpqp program

        .. math::

            \max_{x, \theta, \lambda, s, t} \quad t

        .. math::
            \begin{align*}
                Qx + H \theta + (A_{A_i})^T \lambda_{A_i} + c &= 0\\
                A_{A_i}x - b_ai-F_{a_i}\theta &= 0\\
                A_{A_j}x - b_{A_j}-F_{A_j}\theta + s{j_k} &= 0\\
               t*e_1 &\leq \lambda_{A_i}\\
               t*e_2 &\leq s_{J_i}\\
               t &\geq 0\\
               \lambda_{A_i} &\geq 0\\
               s_{J_i} &\geq 0\\
               A_t\theta &\leq b_t
            \end{align*}

        :param active_set: active set being considered in the optimality test
        :return: dictionary of parameters, or None if active set is not optimal
        """

        zeros = lambda x, y: numpy.zeros((x, y))

        num_x = self.num_x()
        num_constraints = self.num_constraints()
        num_active = len(active_set)
        num_theta_c = self.A_t.shape[0]
        num_activated = len(active_set) - len(self.equality_indices)

        inactive = [i for i in range(num_constraints) if i not in active_set]

        num_inactive = num_constraints - num_active

        num_theta = self.num_t()

        # this will be used to build the optimality expression
        A_list = list()
        b_list = list()

        # 1) Qu + H theta + (A_Ai)^T lambda_Ai + c = 0
        # if num_active > 0:
        # A_list.append([program.Q, zeros(num_x, num_theta), program.A[equality_indices].T, zeros(num_x, num_inactive), zeros(num_x, 1)])
        A_list.append([self.Q, self.H, self.A[active_set].T, zeros(num_x, num_inactive), zeros(num_x, 1)])
        b_list.append([-self.c])
        # 2) A_Ai*u - b_ai-F_ai*theta = 0
        A_list.append([self.A[active_set], -self.F[active_set], zeros(num_active, num_constraints + 1)])
        b_list.append([self.b[active_set]])
        # 3) A_Aj*u - b_aj-F_aj*theta + sj_k= 0
        A_list.append(
            [self.A[inactive], -self.F[inactive], zeros(num_inactive, num_active), numpy.eye(num_inactive),
             zeros(num_inactive, 1)])
        b_list.append([self.b[inactive]])
        # 4) t*e_1 <= lambda_Ai
        # edited on 2/19/2021 to remove the positivity constraint on the equality constraints
        if num_activated >= 0:
            A_list.append(
                [zeros(num_activated, num_x + num_theta + num_active - num_activated), -numpy.eye(num_activated),
                 zeros(num_activated, num_inactive), numpy.ones((num_activated, 1))])
            # A_list.append([zeros(num_active, num_x + num_theta), -numpy.eye(num_active), zeros(num_active, num_inactive),numpy.ones((num_active, 1))])
            b_list.append([zeros(num_activated, 1)])
            # b_list.append([zeros(num_active, 1)])
        # 5) t*e_2 <= s_Ji
        A_list.append([zeros(num_inactive, num_x + num_theta + num_active), -numpy.eye(num_inactive),
                       numpy.ones((num_inactive, 1))])
        b_list.append([zeros(num_inactive, 1)])
        # 6) t >= 0
        t_row = zeros(1, num_x + num_theta + num_constraints + 1)
        t_row[0][-1] = -1
        A_list.append([t_row])
        b_list.append([numpy.array([[0]])])
        # 7) lambda_Ai>= 0
        if num_activated >= 0:
            # edited on 2/19/2021 to remove the positivity constraint on the equality constraints
            A_list.append(
                [zeros(num_activated, num_x + num_theta + num_active - num_activated), -numpy.eye(num_activated),
                 zeros(num_activated, num_inactive + 1)])
            # A_list.append([zeros(num_active, num_x + num_theta), -numpy.eye(num_active), zeros(num_active, num_inactive + 1)])
            b_list.append([zeros(num_activated, 1)])
            # b_list.append([zeros(num_active, 1)])
        # 8) s_Ji>=0
        A_list.append(
            [zeros(num_inactive, num_x + num_theta + num_active), -numpy.eye(num_inactive), zeros(num_inactive, 1)])
        b_list.append([zeros(num_inactive, 1)])
        # 9) A_t*theta<= b_t
        A_list.append([zeros(num_theta_c, num_x), self.A_t, zeros(num_theta_c, num_constraints + 1)])
        b_list.append([self.b_t])

        A_list = [i for i in list(map(remove_size_zero_matrices, A_list)) if i != []]
        b_list = [i for i in list(map(remove_size_zero_matrices, b_list)) if i != []]

        A = ppopt_block(A_list)
        b = ppopt_block(b_list)
        c = make_column(t_row.T)

        lp_active_limit = num_x + num_constraints

        if num_active == 0:
            lp_active_limit = num_constraints

        equality_indices = [i for i in range(0, lp_active_limit)]

        sol = self.solver.solve_lp(c, A, b, equality_indices)

        if sol is not None:
            # separate out the x|theta|lambda|slack|t
            theta_offset = self.Q.shape[0]
            lambda_offset = theta_offset + self.F.shape[1]
            slacks_offset = lambda_offset + num_active
            t_offset = slacks_offset + num_inactive
            return {"x": sol.sol[0:theta_offset],
                    'theta': sol.sol[theta_offset: lambda_offset],
                    'lambda': sol.sol[lambda_offset: slacks_offset],
                    'slack': sol.sol[slacks_offset: t_offset],
                    't': sol.sol[-1],
                    'equality_indices': active_set}
        return None
