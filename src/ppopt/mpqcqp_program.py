from typing import List, Optional, Tuple

import numpy
import sympy

from .mplp_program import MPLP_Program
from .mpqp_program import MPQP_Program
from .nonlinear_critical_region import NonlinearCriticalRegion
from .solver_interface.solver_interface import SolverOutput
from .utils.constraint_utilities import(
    is_full_rank,
    find_redundant_constraints_with_quadratic,
    process_program_constraints,
    find_implicit_equalities,
    generate_reduced_equality_constraints,
)
from .utils.general_utils import (
    latex_matrix,
    make_column,
    ppopt_block,
    remove_size_zero_matrices,
    select_not_in_list,
)
from .utils.symbolic_utils import (
    reduce_redundant_symbolic_constraints,
)

class QConstraint:
    Q: numpy.ndarray
    H: numpy.ndarray
    A: numpy.ndarray
    b: numpy.ndarray
    F: numpy.ndarray
    Q_t: numpy.ndarray

    def __init__(self, Q: numpy.ndarray, H: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, F: numpy.ndarray, Q_t: numpy.ndarray):
        self.Q = Q
        self.H = H
        self.A = A
        self.b = b
        self.F = F
        self.Q_t = Q_t

    def evaluate(self, x: numpy.ndarray, theta: numpy.ndarray) -> float:
        """Evaluates the constraint for a given x and θ."""
        val = x.T @ self.Q @ x + theta.T @ self.H.T @ x + self.A @ x - self.b - self.F @ theta - theta.T @ self.Q_t @ theta
        return float(val[0, 0])
    
    # TODO this does basically the same as evaluate, but for symbolics. Maybe we can combine the two into one method and do a type check to determine the output type?
    def evaluate_symbolic(self, x: sympy.Matrix, theta: sympy.Matrix) -> sympy.Matrix:
        """Evaluates the constraint for a given x and θ symbolically."""
        return x.T @ self.Q @ x + theta.T @ self.H.T @ x + self.A @ x - self.b - self.F @ theta - theta.T @ self.Q_t @ theta

    def evaluate_theta(self, theta: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Evaluates the constraint for a given θ, resulting in a deterministic quadratic constraint.
        :param theta: θ input
        :return: Q, A, b matrices of the deterministic constraint xQx + Ax <= b
        """
        Q_det = self.Q
        A_det = (theta.T @ self.H.T) + self.A
        b_det = self.b + self.F @ theta + theta.T @ self.Q_t @ theta
        return Q_det, A_det, b_det
    
    def evaluate_theta_variable(self) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Considers theta to be an optimization variable, builds a corresponding (deterministic) constraint.
        [x theta]^T [Q 0.5 H; 0.5 H^T -Q_t] [x theta] + [A -F] [x theta] <= b
        """
        Q_det = numpy.block([[self.Q, 0.5 * self.H], [0.5 * self.H.T, -self.Q_t]])
        A_det = numpy.block([[self.A, -self.F]])
        b_det = self.b
        return Q_det, A_det, b_det
    
    def is_convex(self) -> bool:
        """Checks if the quadratic constraint is convex."""
        return numpy.all(numpy.linalg.eigvals(self.Q) >= 10 ** -4)
    

class MPQCQP_Program(MPQP_Program):
    r"""
    The standard class for quadratically constrained quadratic multiparametric programming.

    .. math::
        \begin{align}
        \min_x\quad  \frac{1}{2}x^TQx& + \theta^TH^Tx + c^Tx\\
        \text{s.t.}\quad x^TQ_ix + \theta^TH_i^Tx + A_ix &\leq b_i + F_i\theta + \theta^TQ_{\theta,i}\theta\\
        Ax &\leq b + F\theta\\
        A_{eq}x &= b_{eq}\\
        A_\theta \theta &\leq b_\theta\\
        x &\in R^n\\
        \end{align}
    """

    qconstraints: List[QConstraint]

    def __init__(self, A: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray, H: numpy.ndarray, Q: numpy.ndarray,
                 A_t: numpy.ndarray,
                 b_t: numpy.ndarray, F: numpy.ndarray,
                 qconstraints: List[QConstraint], c_c: Optional[numpy.ndarray] = None,
                 c_t: Optional[numpy.ndarray] = None, Q_t: Optional[numpy.ndarray] = None,
                 equality_indices=None, solver=None, post_process=True):
        """Initialized the MPQP_Program."""
        # calls MPQP_Program's constructor to reduce out burden
        self.qconstraints = qconstraints
        super(MPQCQP_Program, self).__init__(A, b, c, H, Q, A_t, b_t, F, c_c, c_t, Q_t, equality_indices, solver,
                                           post_process)

        # calls the constraint processing to remove redundant constraints
        if post_process:
            self.post_process()

    # overload some basic getters to include the quadratic constraints

    def num_constraints(self) -> int:
        return self.A.shape[0] + len(self.qconstraints)
    
    def num_linear_constraints(self) -> int:
        return self.A.shape[0]
    
    def num_quadratic_constraints(self) -> int:
        return len(self.qconstraints)
    
    def num_inequality_constraints(self) -> int:
        return self.num_constraints() - len(self.equality_indices)
    
    def is_convex(self) -> bool:
        """Checks if the program is convex. Assumes that all quadratic constraints are inequalities."""
        return bool(numpy.all(numpy.linalg.eigvals(self.Q) >= 10 ** -4) and all([q.is_convex() for q in self.qconstraints]))

    def evaluate_objective(self, x, theta_point) -> float:
        r"""
        Evaluates the objective of the multiparametric program. for a given x and θ.

        .. math::
            \frac{1}{2}x^TQx + \theta^TH^Tx+c^Tx

        :param x: x input
        :param theta_point: θ input
        :return: Objective function evaluated at x, θ
        """
        obj_val = 0.5 * x.T @ self.Q @ x + theta_point.T @ self.H.T @ x + self.c.T @ x + self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point

        return float(obj_val[0, 0])
    
    def evaluate_objective_symbolic(self, x, theta) -> sympy.Matrix:
        """Evaluates the objective of the multiparametric program symbolically."""
        return 0.5 * x.T @ self.Q @ x + theta.T @ self.H.T @ x + self.c.T @ x + self.c_c + self.c_t.T @ theta + 0.5 * theta.T @ self.Q_t @ theta

    def warnings(self) -> List[str]:
        """Checks the dimensions of the matrices to ensure consistency."""
        warning_list = MPQP_Program.warnings(self)

        # Quadratically Constrained Quadratic Problem specific warnings

        if len(self.equality_indices) > 0:
            if max(self.equality_indices) >= self.num_linear_constraints():
                warning_list.append('Quadratic equality constraints detected.')

        # TODO these are already caught by MPQP_Program, they should be removed and adjusted for the QCQP case
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
            if min(e_values) < 0:
                warning_list.append(f'Non-convex quadratic objective detected, with eigenvalues {e_values}')
            elif min(e_values) < 10 ** -4:
                warning_list.append(f'Possible positive semi-definite nature detected in Q, eigenvalues {e_values}')

        # return warnings
        return warning_list

    def latex(self) -> List[str]:
        """Creates a latex output for the multiparametric problem."""
        # calls the latex function inherited from MPQP to get most of the output
        output = super(MPQCQP_Program, self).latex()

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
        
        # quadratic constraints
        for q in self.qconstraints:
            H_term = ''
            if not numpy.allclose(q.H, numpy.zeros_like(q.H)):
                H_term = " + " + theta_latex + '^{T}' + latex_matrix(q.H) + x_latex
            Q_theta_term = ''
            if not numpy.allclose(q.Q_t, numpy.zeros_like(q.Q_t)):
                Q_theta_term = " + " + theta_latex + '^{T}' + latex_matrix(q.Q_t) + theta_latex

            output.append(f'$${x_latex}^T{latex_matrix(q.Q)}{x_latex} ' + H_term + f' + {latex_matrix(q.A)}{x_latex} \\leq {latex_matrix(q.b)} + {latex_matrix(q.F)}{theta_latex}' + Q_theta_term + '$$')

        return output

    def solve_theta(self, theta_point: numpy.ndarray) -> Optional[SolverOutput]:
        r"""
        Substitutes a particular realization of θ into the multiparametric problem and solves the resulting
        optimization problem.

        .. math::
            \begin{align}
            \tilde{b} &= b + F\theta\\
            \tilde{b}_eq &= b_{eq} + F_{eq}\theta\\
            \tilde{c}^T &= c^T + \theta^T H^T
            \end{align}

        .. math::
            \begin{align}
            \min_{x}\quad  &\frac{1}{2}x^TQx + \tilde{c}^Tx\\
            \text{s.t.} \quad Ax &\leq \tilde{b}\\
            A_{eq}x &= \tilde{b}_{eq}\\
            x &\in \mathbb{R}^n
            \end{align}

        :param theta_point: An uncertainty realization
        :return: The Solver output of the substituted problem, returns None if not solvable
        """

        if not self.valid_parameter_realization(theta_point):
            return None
        
        q_data = [(q.evaluate_theta(theta_point)) for q in self.qconstraints]
        Q_q = [q[0] for q in q_data]
        A_q = numpy.array([q[1] for q in q_data])
        b_q = numpy.array([q[2] for q in q_data])

        sol_obj = self.solver.solve_miqcqp(Q = self.Q, c = (self.H @ theta_point).reshape(-1,1) + self.c, A = self.A,
                                         b = self.b + (self.F @ theta_point).reshape(-1, 1),
                                         Q_q = Q_q, A_q = A_q, b_q = b_q,
                                         equality_constraints = self.equality_indices)

        if sol_obj is not None:
            sol_obj.obj += self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point
            return sol_obj

        return None
    
    def solve_theta_variable(self) -> Optional[SolverOutput]:
        """
        Leaves Theta as an optimization variable, solves the following problem

        define y' = [x^T theta^T]^T

        min [c^T 0]^Ty' + 1/2 y'^T [Q H; H^T 0] y'
        s.t. [A -F]y' <= b
             quadratic constraints

        :return: the Solver output of the substituted problem, returns None if not solvable
        """

        # This is not consistent with the implementation in MPLP_Program, because there, the A_t @ theta <= b_t constraints are not considered, even though they probably should be

        A_prime = numpy.block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.num_x())), self.A_t]])
        c_prime = numpy.block([[self.c], [numpy.zeros((self.num_t(), 1))]])
        Q_prime = numpy.block([[self.Q, 0.5 * self.H], [0.5 * self.H.T, self.Q_t]])
        b_prime = numpy.block([[self.b], [self.b_t]])

        q_data = [(q.evaluate_theta_variable()) for q in self.qconstraints]
        Q_q_prime = [q[0] for q in q_data]
        A_q_prime = numpy.array([q[1] for q in q_data])
        b_q_prime = numpy.array([q[2] for q in q_data])

        return self.solver.solve_miqcqp(Q=Q_prime, c=c_prime, A=A_prime, b=b_prime, Q_q=Q_q_prime, A_q=A_q_prime, b_q=b_q_prime, equality_constraints=self.equality_indices, q_equality_constraints=[], get_duals=False)


    def optimal_control_law(self, active_set: List[int]) -> Tuple:
        r"""
        This function calculates the optimal control law corresponding to an active set combination. This is effectively
        just manipulating the stationarity conditions and active constraints for x, and λ

        .. math::

            Qx + c + H\theta + \hat{A}^T\hat{\lambda} = 0

            \hat{A}x = \hat{b} + \hat{F}\theta

        .. math::

            \begin{align*}
            x^*(\theta) &= A_x\theta + b_x\\
            \lambda^*(\theta) &= A_l\theta + b_l\\
            \end{align*}

        :param active_set: an active set combination
        :return: a tuple of the optimal x* and λ* functions in the following form(A_x, b_x, A_l, b_l)
        """

        # if the active set contains only linear constraints, then we can use the QP case and transform it to symbolics
        if len(active_set) == 0 or max(active_set) < self.num_linear_constraints():
            A, b, C, d = super().optimal_control_law(active_set)
            theta_sym = sympy.symbols('theta:' + str(self.num_t()), real=True, finite=True)
            x_star = A @ sympy.Matrix([theta_sym]).T + b
            lambda_star = C @ sympy.Matrix([theta_sym]).T + d
            return [x_star], [lambda_star], [sympy.Rational(1)]
        else:
            # create sympy symbols for variables, theta, lagrange multipliers, (nu and beta)
            x_sym = sympy.symbols('x:' + str(self.num_x()), real=True, finite=True)
            theta_sym = sympy.symbols('theta:' + str(self.num_t()), real=True, finite=True)
            lambda_sym = sympy.symbols('lambda:' + str(len(active_set)), real=True, nonnegative=True)
            nu_sym = sympy.symbols('nu', real=True, positive=True)
            beta_sym = sympy.symbols('beta', real=True, positive=True)

            active_linear_indices = [i for i in active_set if i < self.num_linear_constraints()]
            active_quadratic_indices = [i - self.num_linear_constraints() for i in active_set if i >= self.num_linear_constraints()]
            
            # set up the system of quadratic equations
            equations = []
            # build the stationarity condition step by step for readability for now
            # objective gradient
            stationarity = nu_sym * self.Q @ sympy.Matrix(x_sym) + nu_sym * self.H @ sympy.Matrix(theta_sym) + nu_sym * self.c
            # linear constraint gradients
            if len(active_linear_indices) > 0:
                stationarity += self.A[active_linear_indices].T @ sympy.Matrix(lambda_sym[0:len(active_linear_indices)])
            # quadratic constraint gradients
            quadratic_lambda_index = len(active_linear_indices)
            for i in active_quadratic_indices:
                stationarity += lambda_sym[quadratic_lambda_index] * (self.qconstraints[i].Q + self.qconstraints[i].Q.T) @ sympy.Matrix(x_sym) + lambda_sym[quadratic_lambda_index] * self.qconstraints[i].H @ sympy.Matrix(theta_sym) + self.qconstraints[i].A.T @ sympy.Matrix([lambda_sym[quadratic_lambda_index]])
                quadratic_lambda_index += 1

            equations.append(stationarity)

            # active constraints = 0
            active_linear_constraints = self.A[active_linear_indices] @ sympy.Matrix(x_sym) - self.b[active_linear_indices] - self.F[active_linear_indices] @ sympy.Matrix(theta_sym)
            equations.append(active_linear_constraints)
            active_quadratic_constraints = [self.qconstraints[i].evaluate_symbolic(sympy.Matrix(x_sym), sympy.Matrix(theta_sym)) for i in active_quadratic_indices]
            equations.extend(active_quadratic_constraints)

            # normalize the lagrange multipliers or set nu = 1, depending on convexity
            nonconvex_in_active_set = numpy.any([not self.qconstraints[i].is_convex() for i in active_quadratic_indices])
            if nonconvex_in_active_set:
                normalizing_eq = nu_sym ** 2 + sum(l**2 for l in lambda_sym) - beta_sym ** 2
            else:
                normalizing_eq = nu_sym - 1
                equations = [eq.subs(nu_sym, 1) for eq in equations]
            equations.append(normalizing_eq)

            # solve the system of equations
            solution = sympy.solve(equations, [*x_sym, *lambda_sym, nu_sym], rational=True, simplify=True)

            # extract each set of x, lambda, nu
            num_solutions = len(solution)
            x_sol = [solution[i][0:self.num_x()] for i in range(num_solutions)]
            lambda_sol = [solution[i][self.num_x():-1] for i in range(num_solutions)]
            nu_sol = [solution[i][-1] for i in range(num_solutions)]
            
            # iterate over each solution pair, only keep those for which lambda >= 0 can be satisfied
            keep_indices = []
            for i in range(num_solutions):
                this_lambda = lambda_sol[i]
                this_nu = nu_sol[i]
                if numpy.all([(l > 0) != False for l in this_lambda]) and ((this_nu > 0) != False): # required for the cases that lambda is constant, as the solve call will return [] in that case, which is useless TODO is it? might be able to still use that info
                    implications_lambda = [sympy.solve(l.subs({beta_sym:1}) > 0) for l in this_lambda]
                    implications_nu = sympy.solve(this_nu.subs({beta_sym:1}) > 0)
                    implications = [*implications_lambda, implications_nu]
                    implications_satisfied = numpy.all([imp != False for imp in implications])
                    if implications_satisfied:
                        keep_indices.append(i)
                        for implication in implications:
                            if implication != []:
                                x_sol[i] = [sympy.refine(sympy.factor(sympy.LT(x)), implication) + sympy.refine(sympy.factor(x - sympy.LT(x)), implication) for x in x_sol[i]]
                                lambda_sol[i] = [sympy.refine(sympy.factor(sympy.LT(l)), implication) + sympy.refine(sympy.factor(l - sympy.LT(l)), implication) for l in lambda_sol[i]]
                    
            # return the list of x, lambda, nu tuples
            x_sol = [x_sol[i] for i in keep_indices]
            lambda_sol = [lambda_sol[i] for i in keep_indices]
            nu_sol = [nu_sol[i] for i in keep_indices]

            # try to do one final simplification on the kept solutions
            for i in range(len(x_sol)):
                x_sol[i] = [sympy.simplify(x)for x in x_sol[i]]
                lambda_sol[i] = [sympy.simplify(l) for l in lambda_sol[i]]

            return x_sol, lambda_sol, nu_sol


    def gen_cr_from_active_set(self, active_set: List[int]) -> Optional[List[NonlinearCriticalRegion]]:
        # Get list of inactive constraints
        num_linear_constraints = self.num_linear_constraints()
        num_quadratic_constraints = self.num_quadratic_constraints()

        linear_inactive = [i for i in range(num_linear_constraints) if i not in active_set]
        quadratic_inactive = [i - num_linear_constraints for i in range(num_linear_constraints, num_linear_constraints + num_quadratic_constraints) if i not in active_set]

        # Compute optimal control law for active set
        x_stars, lambda_stars, nu_stars = self.optimal_control_law(active_set)
        # if we don't find a control law, we can't build a critical region (most likely due to lower dimensionality of the CR)
        if x_stars == []:
            return None
        theta_syms = sympy.symbols('theta:' + str(self.num_t()), real=True, finite=True)

        returned_regions = []

        for i in range(len(x_stars)):
            x_star = x_stars[i]
            lambda_star = lambda_stars[i]

        # Insert optimal control law into the inactive constraints to build the critical region        
            bounds_from_linear = self.A[linear_inactive] @ sympy.Matrix(x_star) - self.b[linear_inactive] - self.F[linear_inactive] @ sympy.Matrix(theta_syms)
            bounds_from_quadratic = [q.evaluate_symbolic(sympy.Matrix(x_star), sympy.Matrix(theta_syms)) for q in [self.qconstraints[i] for i in quadratic_inactive]]
            bounds_from_theta = self.A_t @ sympy.Matrix(theta_syms) - self.b_t

            original_region_inequalities = []
            for i in bounds_from_linear:
                original_region_inequalities.append(i <= 0)
            for i in bounds_from_quadratic:
                original_region_inequalities.append(i[0] <= 0)
            for i in bounds_from_theta:
                original_region_inequalities.append(i <= 0)
            for i in sympy.Matrix(lambda_star):
                original_region_inequalities.append(i >= 0)

            index_list = []
            region_inequalities = []
            for idx, ineq in enumerate(original_region_inequalities):
                if ineq != True:
                    region_inequalities.append(ineq)
                    index_list.append(idx)

            if self.is_convex():
                region_inequalities, index_list = reduce_redundant_symbolic_constraints(region_inequalities, index_list)

            # Test full dimensionality of the new critical region

            # classify the remaining constraints
            regular_set = [i for i, _ in enumerate(region_inequalities) if index_list[i] < len(bounds_from_linear) + len(bounds_from_quadratic)]
            omega_set = [i for i, _ in enumerate(region_inequalities) if index_list[i] >= len(bounds_from_linear) + len(bounds_from_quadratic) and index_list[i] < len(bounds_from_linear) + len(bounds_from_quadratic) + len(bounds_from_theta)]
            lambda_set = [i for i, _ in enumerate(region_inequalities) if index_list[i] >= len(bounds_from_linear) + len(bounds_from_quadratic) + len(bounds_from_theta)]

            returned_regions.append(NonlinearCriticalRegion(x_star, lambda_star, region_inequalities, active_set, omega_set, lambda_set, regular_set))

        return returned_regions


    def base_constraint_processing(self):
        # This is identical to the MPLP case, with the exception that we don't scale constraints, as some issues came up with that for the quadratically constrained case (even though we only scale linear ones)

        # ensure that all of the equality constraints are at the top
        if len(self.equality_indices) != 0:
            # move all equality constraints to the top
            self.A = numpy.block([[self.A[self.equality_indices]], [select_not_in_list(self.A, self.equality_indices)]])
            self.b = numpy.block([[self.b[self.equality_indices]], [select_not_in_list(self.b, self.equality_indices)]])
            self.F = numpy.block([[self.F[self.equality_indices]], [select_not_in_list(self.F, self.equality_indices)]])

            self.equality_indices = list(range(len(self.equality_indices)))

        self.constraint_datatype_conversion()

        # TODO: add check for a purly parametric equality e.g. c^T theta = b in the main constraint body
        self.A, self.b, self.F, self.A_t, self.b_t = process_program_constraints(self.A, self.b, self.F, self.A_t,
                                                                                self.b_t)
        # we can scale constraints after moving nonzero rows
        # self.scale_constraints()

        # find implicit inequalities in the main constraint body, add them to the equality constraint set
        self.A, self.b, self.F, self.equality_indices = find_implicit_equalities(self.A, self.b, self.F,
                                                                                self.equality_indices)

        # in the case of equality constraints, there can be cases where the constraints are redundant w.r.t. each other
        self.A, self.b, self.F, self.equality_indices = generate_reduced_equality_constraints(self.A, self.b, self.F,
                                                                                            self.equality_indices)


    def process_constraints(self) -> None:
        """Removes redundant constraints from the multiparametric programming problem."""
        
        # First step: run processing on only the linear constraints, if there are any linear constraints that are within themselves redundant, this will reduce the number of QCQPs we have to solve next
        super().process_constraints()

        # Now we build the expressions using the remaining linear and the quadratic constraints
        # form a polytope P := {(x, theta) in R^K : Ax <= b + F theta and A_t theta <= b_t}
        problem_A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
        problem_b = ppopt_block([[self.b], [self.b_t]])

        problem_Q_q = [ppopt_block([[q.Q, 0.5 * q.H], [0.5 * q.H.T, q.Q_t]]) for q in self.qconstraints]
        problem_A_q = numpy.array([ppopt_block([[q.A, -q.F]]) for q in self.qconstraints])
        problem_b_q = numpy.array([q.b for q in self.qconstraints])

        saved_indices = find_redundant_constraints_with_quadratic(problem_A, problem_b, problem_Q_q, problem_A_q,
                                                                   problem_b_q, self.equality_indices, solver=self.solver.solvers['miqcqp'])

        # calculate the indices in the main body and parametric constraints
        saved_linear = [x for x in saved_indices if x < self.num_linear_constraints()]
        saved_theta = [x - self.num_linear_constraints() for x in saved_indices if x >= self.num_linear_constraints() and x < self.num_linear_constraints() + self.A_t.shape[0]]
        saved_quadratic = [x - self.num_linear_constraints() - self.A_t.shape[0] for x in saved_indices if x >= self.num_linear_constraints() + self.A_t.shape[0]]

        # remove redundant linear constraints
        self.A = self.A[saved_linear]
        self.F = self.F[saved_linear]
        self.b = self.b[saved_linear]

        # remove redundant constraints from the parametric constraints
        self.A_t = self.A_t[saved_theta]
        self.b_t = self.b_t[saved_theta]

        # remove redundant quadratic constraints
        self.qconstraints = [self.qconstraints[i] for i in saved_quadratic]


    def check_optimality(self, active_set: list):
        r"""
        Tests if the active set is optimal for the provided mpqcqp program

        .. math::

            \max_{x, \theta, \lambda, s, t, \nu, \beta} \quad t

        .. math::
            \begin{align*}
                \nu Qx + \lambda_{A_i} Q_c x + \nu H \theta + \lambda_{A_i} H_c \theta + (A_{A_i})^T \lambda_{A_i} + \nu c &= 0\\
                A_{A_i}x - b_ai-F_{A_i}\theta &= 0\\
                A_{A_j}x - b_{A_j}-F_{A_j}\theta + s_{j_k} &= 0\\
                x^TQ_{A_i}x + \theta^TH_{A_i}^Tx + A_{A_i}x - b_{A_i} - F_{A_i}\theta + \theta^TQ_{\theta,A_i}\theta &= 0\\
                x^TQ_{A_j}x + \theta^TH_{A_j}^Tx + A_{A_j}x - b_{A_j} - F_{A_j}\theta + \theta^TQ_{\theta,A_j}\theta + s_{j_k}&= 0\\
                \nu^2 + \lambda^T \lambda - \beta^2 = 0\\
               t*e_1 &\leq \lambda_{A_i}\\
               t*e_2 &\leq s_{J_i}\\
               t &\geq 0\\
               \lambda_{A_i} &\geq 0\\
               \nu &\ge 0\\
               s_{J_i} &\geq 0\\
               A_t\theta &\leq b_t
            \end{align*}

        :param active_set: active set being considered in the optimality test
        :return: dictionary of parameters, or None if active set is not optimal
        """

        # IMPLICIT ASSUMPTION HERE BASED ON HOW THIS FUNCTION IS CALLED
        # active_set always begins with the indices of the equality constraints

        # everything for pure linear constraints can be reused from the QP case and extended with more 0s for the quadratic constraint lambdas and slacks
        # we use FJ conditions for the general non convex case, if convex, set nu to 1 which results in KKT --> leave handling up to subsolver
        # for the quadratic constraints, we need to add extra quadratic constraint terms

        zeros = lambda x, y: numpy.zeros((x, y))

        # helper function to convert a 1d array to a row vector, which is required for some ppopt_block uses
        to_row = lambda x: x.reshape(1, -1)

        num_x = self.num_x()
        num_constraints = self.num_constraints()
        num_linear_constraints = self.num_linear_constraints()
        num_quadratic_constraints = self.num_quadratic_constraints()
        num_active = len(active_set)
        linear_active = [i for i in active_set if i < num_linear_constraints]
        quadratic_active = [i - num_linear_constraints for i in active_set if i >= num_linear_constraints]
        num_linear_active = len(linear_active)
        num_quadratic_active = num_active - num_linear_active
        num_theta_c = self.A_t.shape[0]
        num_activated = len(active_set) - len(self.equality_indices)

        linear_inactive = [i for i in range(num_linear_constraints) if i not in active_set]
        quadratic_inactive = [i - num_linear_constraints for i in range(num_linear_constraints, num_linear_constraints + num_quadratic_constraints) if i not in active_set]

        num_linear_inactive = num_linear_constraints - num_linear_active
        num_quadratic_inactive = num_quadratic_constraints - num_quadratic_active
        num_inactive = num_linear_inactive + num_quadratic_inactive

        num_theta = self.num_t()

        nonconvex_in_active_set = False
        if num_quadratic_active > 0:
            nonconvex_in_active_set = numpy.any([not self.qconstraints[i].is_convex() for i in quadratic_active])

        # this will be used to build the optimality expression
        A_list = []
        b_list = []

        Q_q_list = []
        A_q_list = []
        b_q_list = []

        # 1) (nu * Q + lambda * Q_c)u + (nu * H + lambda * H_c)theta + (A_Ai)^T * lambda_Ai + c * nu = 0

        # assemble quadratic term of each dimension of the lagrangian
        for i in range(num_x):
            tmp = zeros(num_x + num_theta + num_constraints + 3, num_x + num_theta + num_constraints + 3)
            # if quadratic constraints are active, add their terms to the quadratic part of the lagrangian
            if num_quadratic_active > 0:
                lambda_block = ppopt_block([[to_row(self.qconstraints[j].Q[i, :] + self.qconstraints[j].Q[i, :].T), to_row(self.qconstraints[j].H[i, :])] for j in quadratic_active])
                tmp[num_x + num_theta + num_linear_active:num_x + num_theta + num_active, 0:num_x + num_theta] = lambda_block
            # add the quadratic part of the objective to the lagrangian
            nu_row = ppopt_block([to_row(self.Q[i,:]), to_row(self.H[i,:])])
            tmp[-2, 0:num_x + num_theta] = nu_row
            Q_q_list.append(tmp)

        # now, we build the linear terms of the lagrangian
        linear_active_matrix = self.A[linear_active].T
        quadratic_active_matrix = zeros(num_x, num_quadratic_active)
        if num_quadratic_active > 0:
            quadratic_active_matrix = ppopt_block([[self.qconstraints[i].A] for i in quadratic_active]).T # linear terms of quadratic constraints, each row is a separate constraint

        A_q_list.append([ppopt_block([zeros(num_x, num_x + num_theta), linear_active_matrix, quadratic_active_matrix, zeros(num_x, num_inactive + 1), self.c, zeros(num_x, 1)])])
        b_q_list.append([zeros(num_x, 1)])

        # 2) A_Ai*u - b_ai-F_ai*theta = 0
        # zeros for all lambda (linear and quadratic), slacks (linear and quadratic), t, nu, beta
        A_list.append([self.A[linear_active], -self.F[linear_active], zeros(num_linear_active, num_constraints + 3)])
        b_list.append([self.b[linear_active]])
        # 3) A_Aj*u - b_aj-F_aj*theta + sj_k= 0
        A_list.append(
            [self.A[linear_inactive], -self.F[linear_inactive], zeros(num_linear_inactive, num_active), numpy.eye(num_linear_inactive),
             zeros(num_linear_inactive, num_quadratic_inactive + 3)])
        b_list.append([self.b[linear_inactive]])
        # 4) t*e_1 <= lambda_Ai
        if num_activated >= 0:
            A_list.append(
                [zeros(num_activated, num_x + num_theta + num_active - num_activated), -numpy.eye(num_activated),
                 zeros(num_activated, num_inactive), numpy.ones((num_activated, 1)), zeros(num_activated, 2)])
            b_list.append([zeros(num_activated, 1)])
        # 5) t*e_2 <= s_Ji
        A_list.append([zeros(num_inactive, num_x + num_theta + num_active), -numpy.eye(num_inactive),
                       numpy.ones((num_inactive, 1)), zeros(num_inactive, 2)])
        b_list.append([zeros(num_inactive, 1)])
        # 6) t >= 0
        # t is now the third to last variable
        t_row = zeros(1, num_x + num_theta + num_constraints + 3)
        t_row[0][-3] = -1
        A_list.append([t_row])
        b_list.append([numpy.array([[0]])])
        # 7) lambda_Ai>= 0
        if num_activated >= 0:
            A_list.append(
                [zeros(num_activated, num_x + num_theta + num_active - num_activated), -numpy.eye(num_activated),
                 zeros(num_activated, num_inactive + 3)])
            b_list.append([zeros(num_activated, 1)])
        # 8) s_Ji>=0
        A_list.append(
            [zeros(num_inactive, num_x + num_theta + num_active), -numpy.eye(num_inactive), zeros(num_inactive, 3)])
        b_list.append([zeros(num_inactive, 1)])
        # 9) A_t*theta<= b_t
        A_list.append([zeros(num_theta_c, num_x), self.A_t, zeros(num_theta_c, num_constraints + 3)])
        b_list.append([self.b_t])
        # 10) active quadratic constraints
        if num_quadratic_active > 0:
            A_q_active = ppopt_block([[q.A, -q.F, zeros(1, num_constraints + 3)] for q in [self.qconstraints[i] for i in quadratic_active]])
            b_q_active = ppopt_block([q.b for q in [self.qconstraints[i] for i in quadratic_active]])
            for i in quadratic_active:
                q = self.qconstraints[i]
                tmp = zeros(num_x + num_theta + num_constraints + 3, num_x + num_theta + num_constraints + 3)
                tmp[0:num_x, 0:num_x] = q.Q
                tmp[0:num_x, num_x:num_x + num_theta] = 0.5 * q.H
                tmp[num_x:num_x + num_theta, 0:num_x] = 0.5 * q.H.T
                tmp[num_x:num_x + num_theta, num_x:num_x + num_theta] = -q.Q_t
                Q_q_list.append(tmp)
            A_q_list.append([A_q_active])
            b_q_list.append([b_q_active])
        # 11) inactive quadratic constraints
        if num_quadratic_inactive > 0:
            A_q_inactive = ppopt_block([[q.A, -q.F, zeros(1, num_constraints + 3)] for q in [self.qconstraints[i] for i in quadratic_inactive]])
            A_q_inactive[:, num_x+num_theta+num_active+num_linear_inactive:num_x+num_theta+num_constraints] = numpy.eye(num_quadratic_inactive)
            b_q_inactive = ppopt_block([q.b for q in [self.qconstraints[i] for i in quadratic_inactive]])
            for i in quadratic_inactive:
                q = self.qconstraints[i]
                tmp = zeros(num_x + num_theta + num_constraints + 3, num_x + num_theta + num_constraints + 3)
                tmp[0:num_x, 0:num_x] = q.Q
                tmp[0:num_x, num_x:num_x + num_theta] = 0.5 * q.H
                tmp[num_x:num_x + num_theta, 0:num_x] = 0.5 * q.H.T
                tmp[num_x:num_x + num_theta, num_x:num_x + num_theta] = -q.Q_t
                Q_q_list.append(tmp)
            A_q_list.append([A_q_inactive])
            b_q_list.append([b_q_inactive.reshape((-1, 1))])
        # 12) nu^2 + lambda^T lambda - beta^2 = 0, beta >= 0
        if nonconvex_in_active_set:
            tmp = zeros(num_x + num_theta + num_constraints + 3, num_x + num_theta + num_constraints + 3)
            tmp[-1][-1] = -1
            tmp[-2][-2] = 1
            lambda_indices = numpy.arange(num_x + num_theta, num_x + num_theta + num_activated)
            tmp[lambda_indices, lambda_indices] = 1
            Q_q_list.append(tmp)
            A_q_list.append([zeros(1, num_x + num_theta + num_constraints + 3)])
            b_q_list.append([zeros(1, 1)])
            A_list.append([zeros(1, num_x + num_theta + num_constraints + 2), -numpy.eye(1)])
            b_list.append([zeros(1, 1)])
        # 13) nu >= 0
        A_list.append([zeros(1, num_x + num_theta + num_constraints + 1), -numpy.eye(1), zeros(1, 1)])
        if nonconvex_in_active_set:
            nu_zero_threshold = zeros(1, 1)
            nu_zero_threshold[0][0] = 10**(-3)
            b_list.append([-nu_zero_threshold])
        else:
            b_list.append([-numpy.ones((1, 1))])

        A_list = [i for i in list(map(remove_size_zero_matrices, A_list)) if i != []]
        b_list = [i for i in list(map(remove_size_zero_matrices, b_list)) if i != []]

        A = ppopt_block(A_list)
        b = ppopt_block(b_list)
        c = make_column(t_row.T)
        A_q = ppopt_block(A_q_list)
        b_q = ppopt_block(b_q_list)

        linear_equality_indices = list(range(0, num_linear_constraints))
        if not nonconvex_in_active_set:
            linear_equality_indices.append(A.shape[0] - 1) # if convex, we add the constraint -nu = -1 to the equality list
            quadratic_equality_indices = list(range(0, num_x + num_quadratic_constraints))
        else:
            quadratic_equality_indices = list(range(0, num_x + num_quadratic_constraints + 1))

        sol = self.solver.solve_miqcqp(None, c, A, b, Q_q_list, A_q, b_q, linear_equality_indices, quadratic_equality_indices, get_duals=False, verbose=False)

        if sol is not None:
            # separate out the x|theta|lambda|slack|t|nu|beta
            theta_offset = num_x
            lambda_offset = theta_offset + num_theta
            slacks_offset = lambda_offset + num_active
            t_offset = slacks_offset + num_inactive
            return {"x": sol.sol[0:theta_offset],
                    'theta': sol.sol[theta_offset: lambda_offset],
                    'lambda': sol.sol[lambda_offset: slacks_offset],
                    'slack': sol.sol[slacks_offset: t_offset],
                    't': sol.sol[-3],
                    'nu': sol.sol[-2],
                    'beta': sol.sol[-1],
                    'equality_indices': active_set}
        return None

    def check_feasibility(self, active_set: List[int], check_rank=True) -> bool:
        r"""
        Checks the feasibility of an active set combination w.r.t. a multiparametric program.

        .. math::
            \begin{align}
            \min_{x,\theta} \quad \quad &0\\
            \text{s.t.}\quad Ax &\leq b + F\theta\\
            A_{i}x &= b_{i} + F_{i}\theta, \quad \forall i \in \mathcal{A}\\
            x^TQ_jx + \theta^TH_j^Tx + A_jx &\leq b_j + F_j\theta + \theta^TQ_{\theta,j}\theta\\
            x^TQ_ix + \theta^TH_i^Tx + A_ix &= b_i + F_i\theta + \theta^TQ_{\theta,i}\theta, \quad \forall i \in \mathcal{A}\\
            A_\theta \theta &\leq b_\theta\\
            x &\in R^n\\
            \theta &\in R^m
            \end{align}

        :param active_set: an active set combination
        :param check_rank: Checks the rank of the LHS matrix for a violation of LINQ if True (default)
        :return: True if active set is feasible else False
        """
        quadratic_offset = self.num_linear_constraints()
        active_linear_constraints = [i for i in active_set if i < quadratic_offset]
        active_quadratic_constraints = [i-quadratic_offset for i in active_set if i >= quadratic_offset]

        if check_rank:
            if not is_full_rank(self.A, active_linear_constraints):
                return False

        # linear constraints can be treated just as before
        A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.num_x())), self.A_t]])
        b = ppopt_block([[self.b], [self.b_t]])
        c = numpy.zeros((self.num_x() + self.num_t(), 1))

        # quadratic constraint handling
        Q_q = [ppopt_block([[q.Q, 0.5 * q.H], [0.5 * q.H.T, q.Q_t]]) for q in self.qconstraints]
        A_q = numpy.array([ppopt_block([[q.A, -q.F]]) for q in self.qconstraints])
        b_q = numpy.array([q.b for q in self.qconstraints])

        return self.solver.solve_miqcqp(None, c, A, b, Q_q, A_q, b_q, active_linear_constraints, active_quadratic_constraints, get_duals=False) is not None