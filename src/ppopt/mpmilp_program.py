from dataclasses import dataclass
from typing import Union, List, Optional

import numpy

from .mplp_program import MPLP_Program
from .solver import Solver
from .solver_interface.solver_interface_utils import SolverOutput
from .utils.constraint_utilities import detect_implicit_equalities, find_redundant_constraints
from .utils.general_utils import ppopt_block


class MPMILP_Program(MPLP_Program):
    # noinspection SpellCheckingInspection
    r"""
        The standard class for linear multiparametric programming
        .. math::
            \min \theta^TH^Tx + c^Tx + c_c + c_t^T\theta + \frac{1}{2}\theta^TQ_t\theta
        .. math::
            \begin{align}
            A_{eq}x &= b_{eq} + F_{eq}\theta\\
            Ax &\leq b + F\theta\\
            A_\theta \theta &\leq b_\theta\\
            x_i &\in \mathbb{R} \lxor \{0,1\}\\
            \end{align}

        Equality constraints containing only binary variables cannot also be parametric, as that generate a non-convex and
        discrete feasible parameter space
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

    binary_indices: Union[List[int], numpy.ndarray]
    cont_indices: List[int]

    solver: Solver = Solver()

    def __init__(self, A: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray, H: numpy.ndarray, A_t: numpy.ndarray,
                 b_t: numpy.ndarray, F: numpy.ndarray, binary_indices=None, c_c: Optional[numpy.ndarray] = None,
                 c_t: Optional[numpy.ndarray] = None, Q_t: Optional[numpy.ndarray] = None,
                 equality_indices: List[int] = None,
                 solver: Solver = Solver(), ):
        """Initializes the MPMILP_Program"""
        super().__init__(A, b, c, H, A_t, b_t, F, c_c, c_t, Q_t, equality_indices, solver)
        self.binary_indices = binary_indices
        self.cont_indices = [i for i in range(self.num_x()) if i not in self.binary_indices]

        if len(self.cont_indices) == 0:
            print("Pure Integer case is not considered here only the Mixed case!!!")

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

        # now we call the process constraints routine to polish the constraints before we move to solving
        self.process_constraints()

    def evaluate_objective(self, x: numpy.ndarray, theta_point: numpy.ndarray):
        """Evaluates the objective f(x,theta)"""
        return theta_point.T @ self.H.T @ x + self.c.T @ x + self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point

    def process_constraints(self, find_implicit_equalities=True) -> None:
        """
        This is the constraint processing function for the mixed integer multiparametric case, this is separate from
        the continuous case as we need to use milps to remove the strongly redundant constraints. Doesn't support
        weakly redundant constraint removal but that is taken care of via when the integers are substituted

        :param find_implicit_equalities: Will find trivial instances where inequality constraints generate an equality constraint
        :return: None
        """
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

            # put active constraints on the top
            self.A = ppopt_block([[A_eq], [A_ineq]])
            self.b = ppopt_block([[b_eq], [b_ineq]])
            self.F = ppopt_block([[F_eq], [F_ineq]])

            # update problem active set
            self.equality_indices = [i for i in range(len(temp_active_set))]

        # recalculate bc we have moved everything around
        problem_A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
        problem_b = ppopt_block([[self.b], [self.b_t]])

        # find the milp redundant constraints
        saved_indices = []
        for i in range(self.num_inequality_constraints()):
            new_eq_set = [*self.equality_indices, i + self.num_equality_constraints()]
            sol = self.solver.solve_milp(None, problem_A, problem_b, new_eq_set, bin_vars=self.binary_indices)
            if sol is not None:
                saved_indices.append(i + self.num_equality_constraints())

        saved_upper = [i for i in saved_indices if i < self.A.shape[0]]

        self.A = self.A[saved_upper]
        self.F = self.F[saved_upper]
        self.b = self.b[saved_upper]

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

        # find any integer only constraints and ignore them
        kept_constraints = []
        for i in range(self.num_constraints()):

            # constraint of the type sum(a_i*y_i, i in I) ?? b -> we do not need this
            if numpy.allclose(A_cont[i], 0 * A_cont[i]) and numpy.allclose(self.F[i], 0 * self.F[i]):
                continue
            kept_constraints.append(i)

        # remove integer only constraints from equality set
        equality_set = [i for i in self.equality_indices if i in kept_constraints]

        A_cont = A_cont[kept_constraints]
        A_bin = A_bin[kept_constraints]
        b = self.b[kept_constraints] - A_bin @ fixed_combination
        F = self.F[kept_constraints]

        c = self.c[self.cont_indices]
        c_c = self.c_c + self.c[self.binary_indices].T @ fixed_combination
        H_c = self.H[self.cont_indices]
        H_d = self.H[self.binary_indices]

        c_t = self.c_t + fixed_combination.T @ H_d

        sub_problem = MPLP_Program(A_cont, b, c, H_c, self.A_t, self.b_t, F, c_c, c_t, self.Q_t, equality_set,
                                   self.solver)
        sub_problem.process_constraints(True)
        return sub_problem

    def solve_theta(self, theta_point: numpy.ndarray, deterministic_solver='gurobi') -> Optional[SolverOutput]:
        """
        Solves the substituted problem,with the provided theta

        :param theta_point:
        :param deterministic_solver:
        :return:
        """
        return self.solver.solve_milp(self.c + self.H @ theta_point, self.A, self.b + self.F @ theta_point,
                                      self.equality_indices, self.binary_indices)

    def check_bin_feasibility(self, partial_fixed_bins: List = None) -> bool:
        """
        Checks if a partial binary substitution is feasible in the MILP sense

        if we have the following binary variables [x2, x3, x7 ,x9] and we pass the partial fix [1, 0] to this problem
        then it will check the feasibility of the constraint set with x2 = 1 and x3 = 0

        :param partial_fixed_bins: a set of values to fix binary variable with
        :return: True of feasible, False otherwise
        """
        if partial_fixed_bins is None:
            partial_fixed_bins = []

        new_equ_rows_A = []
        new_equ_rows_b = []

        for i in range(len(partial_fixed_bins)):
            new_row = [0 for _ in range(self.num_x() + self.num_t())]
            new_row[self.binary_indices[i]] = 1
            new_equ_rows_A.append(new_row)
            new_equ_rows_b.append(partial_fixed_bins[i])

        new_equ_rows_A = numpy.array(new_equ_rows_A)
        new_equ_rows_b = numpy.array(new_equ_rows_b).reshape(-1, 1)

        # recalculate bc we have moved everything around
        problem_A = ppopt_block(
            [[new_equ_rows_A], [self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
        problem_b = ppopt_block([[new_equ_rows_b], [self.b], [self.b_t]])

        eq = [*list(range(len(partial_fixed_bins))), *[i + len(partial_fixed_bins) for i in self.equality_indices]]
        return self.solver.solve_milp(None, problem_A, problem_b, eq, bin_vars=self.binary_indices) is not None
