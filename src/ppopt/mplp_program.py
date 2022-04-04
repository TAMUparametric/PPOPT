from dataclasses import dataclass
from typing import List, Union, Optional, Tuple

import numpy

from .solver import Solver
from .solver_interface.solver_interface_utils import SolverOutput
from .utils.chebyshev_ball import chebyshev_ball
from .utils.constraint_utilities import constraint_norm, is_full_rank, \
    detect_implicit_equalities, find_redundant_constraints
from .utils.general_utils import make_column, latex_matrix, select_not_in_list, ppopt_block, remove_size_zero_matrices


def calc_weakly_redundant(A, b, equality_set: List[int] = None, deterministic_solver='gurobi'):
    if equality_set is None:
        equality_set = []

    kept_indices = []
    for i in range(len(equality_set), A.shape[0]):
        sol = chebyshev_ball(A, b, [*equality_set, i], deterministic_solver=deterministic_solver)
        if sol is not None:
            if sol.sol[-1] >= 1e-12:
                kept_indices.append(i)

    return [*equality_set, *kept_indices]


# noinspection GrazieInspection
@dataclass
class MPLP_Program:
    r"""
    The standard class for linear multiparametric programming
    .. math::
        \min \theta^TH^Tx + c^Tx
    .. math::
        \begin{align}
        Ax &\leq b + F\theta\\
        A_{eq}x &= b_{eq}\\
        A_\theta \theta &\leq b_\theta\\
        x &\in R^n\\
        \end{align}

    """

    # uses dataclass to create the __init__  with post-processing in the __post_init__
    # member variables of the MPLP_Program class

    A: numpy.ndarray
    b: numpy.ndarray
    c: numpy.ndarray
    H: numpy.ndarray
    A_t: numpy.ndarray
    b_t: numpy.ndarray
    F: numpy.ndarray
    c_c: numpy.ndarray
    c_t: numpy.ndarray
    Q_t: numpy.ndarray

    equality_indices: Union[List[int], numpy.ndarray]

    solver: Solver = Solver()

    def __init__(self, A, b, c, H, A_t, b_t, F, c_c=None, c_t=None, Q_t=None, equality_indices=None, solver=None):

        self.A = A
        self.b = b
        self.c = c
        self.H = H
        self.A_t = A_t
        self.b_t = b_t
        self.F = F

        if c_c is None:
            c_c = numpy.array([[0.0]])
        self.c_c = c_c

        if c_t is None:
            c_t = numpy.zeros((self.num_t(), 1))
        self.c_t = c_t

        if Q_t is None:
            Q_t = numpy.zeros((self.num_t(), self.num_t()))
        self.Q_t = Q_t

        if equality_indices is None:
            equality_indices = []

        self.equality_indices = equality_indices

        if solver is None:
            solver = Solver()

        self.solver = solver

    def __post_init__(self):
        """Called after __init__ this is used as a post-processing step after the dataclass generated __init__."""
        if self.equality_indices is None:
            self.equality_indices = []

        if len(self.equality_indices) != 0:
            # move all equality constraints to the top
            self.A = numpy.block(
                [[self.A[self.equality_indices]], [numpy.delete(self.A, self.equality_indices, axis=0)]])
            self.b = numpy.block(
                [[self.b[self.equality_indices]], [numpy.delete(self.b, self.equality_indices, axis=0)]])
            self.F = numpy.block(
                [[self.F[self.equality_indices]], [numpy.delete(self.F, self.equality_indices, axis=0)]])
            # reassign the equality constraint indices to the top indices after move
            self.equality_indices = [i for i in range(len(self.equality_indices))]

        # ensures that
        self.warnings()
        self.process_constraints(find_implicit_equalities=True)

    def num_x(self) -> int:
        """Returns number of parameters."""
        return self.A.shape[1]

    def num_t(self) -> int:
        """Returns number of uncertain variables."""
        return self.F.shape[1]

    def num_constraints(self) -> int:
        """Returns number of constraints."""
        return self.A.shape[0]

    def num_inequality_constraints(self) -> int:
        return self.A.shape[0] - len(self.equality_indices)

    def num_equality_constraints(self) -> int:
        return len(self.equality_indices)

    def evaluate_objective(self, x: numpy.ndarray, theta_point: numpy.ndarray):
        return theta_point.T @ self.H.T @ x + self.c.T @ x + self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point

    def warnings(self) -> List[str]:
        """Checks the dimensions of the matrices to ensure consistency."""
        warning_list = list()

        # check if b is a column vector
        if len(self.b.shape) != 2:
            warning_list.append(f'The b matrix is not a column vector b{self.b.shape}')
            self.b = make_column(self.b)
            warning_list.append('This has been corrected')

        # check if c is a column matrix
        if len(self.c.shape) != 2:
            warning_list.append(f'The c vector is not a column vector c{self.c.shape}')
            self.c = make_column(self.c)
            warning_list.append('This has been corrected')

        # check if c and A have consistent dimensions
        if self.A.shape[1] != self.c.shape[0]:
            warning_list.append(
                f'The A and b matrices disagree in number of parameters A{self.A.shape}, c{self.c.shape}')

        # check is A and b agree with each other
        if self.A.shape[0] != self.b.shape[0]:
            warning_list.append(f'The A and b matrices disagree in vertical dimension A{self.A.shape}, b{self.b.shape}')

        # check is A and b agree with each other
        if self.A_t.shape[0] != self.b_t.shape[0]:
            warning_list.append(
                f'The A and b matrices disagree in vertical dimension A{self.A_t.shape}, b{self.b_t.shape}')

        # check dimensions of A and F matrix
        if self.A.shape[0] != self.F.shape[0]:
            warning_list.append(
                f"The A and F matrices disagree in vertical dimension A{self.A.shape}, F {self.F.shape}")

        # return warnings
        return warning_list

    # Checks warnings again and prints warnings
    def display_warnings(self) -> None:
        """Displaces warnings."""
        print(self.warnings())

    def display_latex(self) -> None:
        """Displaces Latex text of the multiparametric problem."""
        output = self.latex()
        for i in output:
            print(i)

    def latex(self) -> List[str]:
        """
        Generates latex of the multiparametric problem

        :return: returns latex of the
        """
        output = list()

        # create string variables for x and theta
        x = ['x_{' + f'{i}' + '}' for i in range(self.num_x())]
        theta = ['\\theta_{' + f'{i}' + '}' for i in range(self.num_t())]

        # create the latex matrices that represent x and theta
        # using the latex_matrix function from utils.general_utils
        x_latex = latex_matrix(x)
        theta_latex = latex_matrix(theta)

        # builds the objective latex
        added_term = ''
        if not numpy.allclose(self.H, numpy.zeros_like(self.H)):
            added_term = " + " + theta_latex + '^{T}' + latex_matrix(self.H) + x_latex

        obj = "$$" + "\\min_{x}" + latex_matrix(self.c) + "^T" + x_latex + added_term + "$$"

        output.append(obj)

        # adds the inequality constraint latex if applicable
        if self.num_constraints() - len(self.equality_indices) > 0:
            A_ineq = latex_matrix(select_not_in_list(self.A, self.equality_indices))
            b_ineq = latex_matrix(select_not_in_list(self.b, self.equality_indices))
            F_ineq = latex_matrix(select_not_in_list(self.F, self.equality_indices))
            output.append("$$" + ''.join([A_ineq, x_latex, '\\leq', b_ineq, '+', F_ineq, theta_latex]) + "$$")

        # adds the equality constraint latex if applicable
        if len(self.equality_indices) > 0:
            A_eq = latex_matrix(self.A[self.equality_indices])
            b_eq = latex_matrix(self.b[self.equality_indices])
            F_eq = latex_matrix(self.F[self.equality_indices])
            output.append("$$" + ''.join([A_eq, x_latex, '=', b_eq, '+', F_eq, theta_latex]) + "$$")

        # adds the theta constraint latex
        output.append("$$" + latex_matrix(self.A_t) + theta_latex + '\\leq' + latex_matrix(self.b_t) + "$$")

        return output

    def scale_constraints(self) -> None:
        """Rescales the constraints of the multiparametric problem to ||[A|-F]||_i = 1, in the L2 sense."""
        # scale the [A| b, F] constraint by the H = [A|-F] rows
        H = numpy.block([self.A, -self.F])
        norm = constraint_norm(H)
        self.A = self.A / norm
        self.b = self.b / norm
        self.F = self.F / norm

        # scale the A_t constraint by the norm of it's rows
        norm = constraint_norm(self.A_t)
        self.A_t = self.A_t / norm
        self.b_t = self.b_t / norm

    def process_constraints(self, find_implicit_equalities=True) -> None:
        """Removes redundant constraints from the multiparametric programming problem."""
        self.constraint_datatype_conversion()
        self.scale_constraints()

        if find_implicit_equalities:
            problem_A = ppopt_block([[self.A, -self.F]])
            problem_b = ppopt_block([[self.b]])

            constraint_pairs = detect_implicit_equalities(problem_A, problem_b)

            keep = [i[0] for i in constraint_pairs]
            remove = [i[1] for i in constraint_pairs]

            keep = list(set(keep))
            keep.sort()

            remove = list(set(remove))
            remove.sort()

            # make sure to only remove the unneeded inequalities -> only for duplicate constraints
            remove = [i for i in remove if i not in keep]

            # our temporary new active set for the problem
            temp_active_set = [*self.equality_indices, *keep]

            # what we are keeping
            survive = lambda x: x not in temp_active_set and x not in remove
            kept_ineqs = [i for i in range(self.num_constraints()) if survive(i)]

            # data marshaling
            A_eq = self.A[temp_active_set]
            b_eq = self.b[temp_active_set]
            F_eq = self.F[temp_active_set]

            A_ineq = self.A[kept_ineqs]
            b_ineq = self.b[kept_ineqs]
            F_ineq = self.F[kept_ineqs]

            self.A = ppopt_block([[A_eq], [A_ineq]])
            self.b = ppopt_block([[b_eq], [b_ineq]])
            self.F = ppopt_block([[F_eq], [F_ineq]])

            # update problem active set
            self.equality_indices = [i for i in range(len(temp_active_set))]

        # recalculate bc we have moved everything around
        problem_A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
        problem_b = ppopt_block([[self.b], [self.b_t]])

        saved_indices = find_redundant_constraints(problem_A, problem_b, self.equality_indices,
                                                   solver=self.solver.solvers['lp'])
        # saved_indices = calculate_redundant_constraints(problem_A, problem_b)

        saved_upper = [i for i in saved_indices if i < self.A.shape[0]]
        # saved_lower = [i - self.A.shape[0] for i in saved_indices if i >= self.A.shape[0]]

        self.A = self.A[saved_upper]
        self.F = self.F[saved_upper]
        self.b = self.b[saved_upper]

        # recalculate bc we have moved everything around
        problem_A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
        problem_b = ppopt_block([[self.b], [self.b_t]])

        # saved_indices = calc_weakly_redundant(problem_A, problem_b, self.equality_indices)
        # saved_indices = calculate_redundant_constraints(problem_A, problem_b)

        saved_upper = [i for i in saved_indices if i < self.A.shape[0]]
        # saved_lower = [i - self.A.shape[0] for i in saved_indices if i >= self.A.shape[0]]

        self.A = self.A[saved_upper]
        self.F = self.F[saved_upper]
        self.b = self.b[saved_upper]

        # print(f'Removed {self.A.shape[0] - len(saved_upper)} Weakly Redundant Constraints')

        self.scale_constraints()

    def constraint_datatype_conversion(self) -> None:
        """
        Makes sure that all the data types of the problem are in fp64, this is important as some solvers do not accept integral data types
        """
        self.A = self.A.astype('float64')
        self.c = self.c.astype('float64')
        self.b = self.b.astype('float64')
        self.F = self.F.astype('float64')
        self.A_t = self.A_t.astype('float64')
        self.b_t = self.b_t.astype('float64')
        self.H = self.H.astype('float64')
        self.c_c = self.c_c.astype('float64')
        self.c_t = self.c_t.astype('float64')
        self.Q_t = self.Q_t.astype('float64')

    def solve_theta(self, theta_point: numpy.ndarray, deterministic_solver='gurobi') -> Optional[SolverOutput]:
        r"""
        Substitutes theta into the multiparametric problem and solves the following optimization problem

        .. math::

            \min_{x} \tilde{c}^Tx

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

        sol_obj = self.solver.solve_lp(c=self.H @ theta_point + self.c, A=self.A, b=self.b + self.F @ theta_point,
                                       equality_constraints=self.equality_indices)

        if sol_obj is not None:
            sol_obj.obj += self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point
            return sol_obj

        return None

    def solve_theta_variable(self) -> Optional[SolverOutput]:
        """
        Leaves Theta as an optimization variable, solves the following problem

        define y' = [x^T theta^T]^T

        min [c^T 0]^Ty'
        s.t. [A -F]y' <= b

        :return: the Solver output of the substituted problem, returns None if not solvable
        """

        A_prime = numpy.block([self.A, -self.F])
        c_prime = numpy.block([[self.c], [numpy.zeros((self.num_t(), 1))]])

        return self.solver.solve_lp(c=c_prime, A=A_prime, b=self.b, equality_constraints=self.equality_indices)

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

        aux = numpy.linalg.pinv(self.A[active_set])

        parameter_A = aux @ self.F[active_set]
        parameter_b = aux @ self.b[active_set]

        lagrange_A = -aux.T @ self.H
        lagrange_b = -aux.T @ self.c

        return parameter_A, parameter_b, lagrange_A, lagrange_b

    # noinspection SpellCheckingInspection
    def check_active_set_rank(self, active_set):
        r"""
        Checks the rank of the matrix is equal to the cardinality of the active set

        .. math::

            \textrm{Rank}(A_{\mathcal{A}}) = |\mathcal{A}|

        :param active_set:
        :return: True if full rank otherwise false
        """
        return is_full_rank(self.A, active_set)

    def check_feasibility(self, active_set, check_rank=True) -> bool:
        r"""
        Checks the feasibility of an active set combination w.r.t. a multiparametric program.

        .. math::

            \min_{x,\theta} 0

        .. math::
            \begin{align}
            Ax &\leq b + F\theta\\
            A_{i}x &= b_{i} + F_{i}\theta, \quad \forall i \in \mathcal{A}\\
            A_\theta \theta &\leq b_\theta\\
            x &\in R^n\\
            \theta &\in R^m
            \end{align}

        :param active_set: an active set
        :param check_rank: Checks the rank of the LHS matrix for a violation of LINQ if True (default)
        :return: True if active set feasible else False
        """

        if check_rank:
            if not is_full_rank(self.A, active_set):
                return False

        A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.num_x())), self.A_t]])
        b = ppopt_block([[self.b], [self.b_t]])
        c = numpy.zeros((self.num_x() + self.num_t(), 1))
        return self.solver.solve_lp(c, A, b, active_set) is not None

    def check_optimality(self, active_set):
        r"""
        Tests if the active set is optimal for the provided mpLP program

        .. math::

            \max_{x, \theta, \lambda, s, t} \quad t

        .. math::
            \begin{align*}
                H \theta + (A_{A_i})^T \lambda_{A_i} + c &= 0\\
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
        if len(active_set) != self.num_x():
            return False

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
        A_list.append([zeros(num_x, num_x), self.H, self.A[active_set].T, zeros(num_x, num_inactive), zeros(num_x, 1)])
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
            theta_offset = self.num_x()
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

    def feasible_theta_point(self) -> Optional[numpy.ndarray]:
        """
        This generates a feasible theta point for the multiparametric problem

        :return:
        """

        sol = self.feasible_space_chebychev_ball()

        if sol is None:
            return None

        return sol.sol[self.num_x(): self.num_x() + self.num_t()].reshape(-1, 1)

    def gen_optimal_active_set(self) -> Optional[List[int]]:
        """
        Self contained method to geometrically sample the theta feasible space to generate an optimal active set.

        :return: an optimal active set
        """

        sol = self.feasible_space_chebychev_ball()

        if sol is None:
            return None

        theta_point = sol.sol[self.num_x(): self.num_x() + self.num_t()].reshape(-1, 1)
        radius = sol.sol[-1]

        max_iter = 500

        for _ in range(max_iter):

            pert = numpy.random.uniform(-radius, radius, (self.num_t(), 1))
            test_point = pert + theta_point

            is_optimal = self.solve_theta(test_point)

            if is_optimal is not None:
                if is_optimal.active_set.size <= self.num_x():
                    return is_optimal.active_set.tolist()

        return None

    def feasible_space_chebychev_ball(self):
        """
        Formulates and solves the (x, \theta) chebychev ball of the multiparametric program.


        :return: makes a che
        """
        A = numpy.block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.num_x())), self.A_t]])
        b = numpy.block([[self.b], [self.b_t]])
        sol = chebyshev_ball(A, b, equality_constraints=self.equality_indices,
                             deterministic_solver=self.solver.solvers['lp'])
        return sol

    def sample_theta_space(self, num_samples: int = 100) -> Optional[list]:
        """
        Samples the theta feasible space with a Diken walk algorithm. This is typically used to initiate the graph
        and geometric algorithm.

        :return: list of found optimal active sets
        """

        sol = self.feasible_space_chebychev_ball()

        if sol is None:
            return None

        theta = sol.sol[self.num_x(): self.num_x() + self.num_t()].reshape(-1, 1)
        radius = sol.sol[-1]
        found_active_sets = []

        def random_direction():
            vec = numpy.random.randn(self.num_t()).reshape(self.num_t(), -1)
            return vec / numpy.linalg.norm(vec, 2)

        for _ in range(num_samples):

            new_theta = theta + numpy.random.rand() * radius * random_direction()
            sol = self.solve_theta(new_theta)

            if sol is not None:
                found_active_sets.append(tuple(sol.active_set.tolist()))
                theta = new_theta

        return [list(active_set) for active_set in set(found_active_sets)]

    # noinspection SpellCheckingInspection
    def gen_feasible_theta_space(self):
        r"""
        Generated the theta feasible space of a multiparametric program (with up to affine constraints)

        this is done by solving the following linear program for each reduced constraint

        min -A_i x

        s.t. Ax \leq F \theta + b

        then the solutions are transformed into the following results

        A' = [-F \theta; A_theta]
        b' = [b - {A_i x} min, b_theta]

        :return: A', b' for A' \theta \leq b'
        """

        # find all of the A_i of the
        # for _ in range(len(self.equality_indices), self.A.shape[0]):
        #     pass
        pass
