import numpy
import pytest
from scipy.stats import random_correlation

from src.ppopt.critical_region import CriticalRegion
from src.ppopt.mpmodel import MPModeler
from src.ppopt.mp_solvers.mpqp_combinatorial import CombinationTester
from src.ppopt.mp_solvers.solve_mpqp import solve_mpqp
from src.ppopt.mplp_program import MPLP_Program
from src.ppopt.mpmilp_program import MPMILP_Program
from src.ppopt.mpmiqp_program import MPMIQP_Program
from src.ppopt.mpqp_program import MPQP_Program
from src.ppopt.solution import Solution
from src.ppopt.utils.general_utils import make_column
from src.ppopt.mpmodel import MPModeler, VariableType


@pytest.fixture()
def qp_problem():
    """The factory problem from the mp book by Richard."""
    A = numpy.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
         [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = 25.0 * make_column([1, 1, 1, 1])
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = 2.0 * numpy.diag([153, 162, 162, 126])

    A_t = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    b_t = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((A.shape[1], F.shape[1]))
    return MPQP_Program(A, b, c, H, Q, A_t, b_t, F)


@pytest.fixture()
def factory_solution():
    """The factory problem from the mp book by Richard."""
    A = numpy.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
         [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = 25.0 * make_column([1, 1, 1, 1])
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = 2.0 * numpy.diag([153, 162, 162, 126])

    A_t = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    b_t = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((A.shape[1], F.shape[1]))
    return solve_mpqp(MPQP_Program(A, b, c, H, Q, A_t, b_t, F))


@pytest.fixture()
def simple_mpqp_problem():
    Q = numpy.array([[1]])
    A = numpy.array([[1], [-1]])
    b = numpy.array([[5], [0]])
    c = numpy.array([[0]])
    F = numpy.array([[1], [1]])
    A_t = numpy.array([[-1], [1]])
    b_t = numpy.array([[0], [1]])
    H = numpy.zeros((F.shape[1], Q.shape[0]))
    prob = MPQP_Program(A, b, c, H, Q, A_t, b_t, F)
    prob.warnings()
    return prob


@pytest.fixture()
def region() -> CriticalRegion:
    """A square critical region with predictable properties."""
    A = numpy.eye(2)
    b = numpy.zeros((2, 1))
    C = numpy.eye(2)
    d = numpy.zeros((2, 1))
    E = numpy.block([[numpy.eye(2)], [-numpy.eye(2)]])
    f = make_column([1, 1, 0, 0])
    return CriticalRegion(A, b, C, d, E, f, [])


@pytest.fixture()
def linear_program() -> MPLP_Program:
    """A simple mplp to test the dimensional correctness of its functions."""
    A = numpy.eye(3)
    b = numpy.zeros((3, 1))
    F = numpy.ones((3, 10))
    A_t = numpy.block([[-numpy.eye(5)], [numpy.eye(5)]])
    b_t = numpy.ones((10, 1))
    c = numpy.ones((3, 1))
    H = numpy.zeros((A.shape[1], F.shape[1]))
    return MPLP_Program(A, b, c, H, A_t, b_t, F, None, None, None, equality_indices=[0])


@pytest.fixture()
def quadratic_program() -> MPQP_Program:
    """A simple mplp to test the dimensional correctness of its functions."""
    A = numpy.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
         [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = 25 * make_column([1, 1, 1, 1])
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = 2.0 * numpy.diag([153, 162, 162, 126])

    CRa = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    CRb = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((F.shape[1], Q.shape[0]))

    prog = MPQP_Program(A, b, c, H, Q, CRa, CRb, F, post_process=False)
    prog.scale_constraints()
    return prog


@pytest.fixture()
def blank_solution():
    """
    Blank solution

    a simple mplp to test the dimensional correctness of its functions.
    """
    A = numpy.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
         [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = 25 * make_column([1, 1, 1, 1])
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = 2.0 * numpy.diag([153, 162, 162, 126])

    CRa = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    CRb = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((F.shape[1], Q.shape[0]))

    prog = MPQP_Program(A, b, c, H, Q, CRa, CRb, F)
    prog.scale_constraints()
    return Solution(prog, [])


@pytest.fixture()
def filled_solution(region):
    """
    Blank solution with the single square region

    a simple mplp to test the dimensional correctness of its functions
    """
    A = numpy.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
         [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = 25 * make_column([1, 1, 1, 1])
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = 2.0 * numpy.diag([153, 162, 162, 126])

    CRa = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    CRb = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((F.shape[1], Q.shape[0]))

    prog = MPQP_Program(A, b, c, H, Q, CRa, CRb, F)
    prog.scale_constraints()
    return Solution(prog, [region])


@pytest.fixture()
def blank_combo_tester():
    """Blank murder list."""
    return CombinationTester()


@pytest.fixture()
def filled_combo_tester():
    """Partially filled murder list."""
    c = CombinationTester()
    c.add_combo([1])
    c.add_combo([2])
    c.add_combo([3])
    c.add_combo([1, 5])
    return c


@pytest.fixture()
def simple_mpMILP():
    """Simple mpMILP to solve for """
    A = numpy.array([[0, 1, 1], [1, 0, 0], [-1, 0, 0], [1, -1, 0], [1, 0, -1]])
    b = numpy.array([1, 0, 0, 0, 0]).reshape(-1, 1)
    F = numpy.array([0, 1, 0, 0, 0]).reshape(-1, 1)
    c = numpy.array([-3, 0, 0]).reshape(-1, 1)
    H = numpy.zeros((F.shape[1], A.shape[1])).T
    A_t = numpy.array([1, 1]).reshape(-1, 1)
    b_t = numpy.array([2, 2]).reshape(-1, 1)

    mpmilp = MPMILP_Program(A, b, c, H, A_t, b_t, F, binary_indices=[1, 2])
    return mpmilp


@pytest.fixture()
def simple_mpLP():
    """Simple mpMILP to solve for """
    A = numpy.array([[0, 1, 1], [1, 0, 0], [-1, 0, 0], [1, -1, 0], [1, 0, -1]])
    b = numpy.array([1, 0, 0, 0, 0]).reshape(-1, 1)
    F = numpy.array([0, 1, 0, 0, 0]).reshape(-1, 1)
    c = numpy.array([-3, 0, 0]).reshape(-1, 1)
    H = numpy.zeros((F.shape[1], A.shape[1])).T
    A_t = numpy.array([1, 1]).reshape(-1, 1)
    b_t = numpy.array([2, 2]).reshape(-1, 1)

    mpmilp = MPLP_Program(A, b, c, H, A_t, b_t, F)
    return mpmilp


@pytest.fixture()
def simple_mpMIQP():
    """Simple mpMILP to solve for """
    A = numpy.array([[0, 1, 1], [1, 0, 0], [-1, 0, 0], [1, -1, 0], [1, 0, -1]])
    b = numpy.array([1, 0, 0, 0, 0]).reshape(-1, 1)
    F = numpy.array([0, 1, 0, 0, 0]).reshape(-1, 1)
    c = numpy.array([-3, 0, 0]).reshape(-1, 1)
    H = numpy.zeros((F.shape[1], A.shape[1])).T
    Q = numpy.eye(3)
    A_t = numpy.array([1, 1]).reshape(-1, 1)
    b_t = numpy.array([2, 2]).reshape(-1, 1)

    mpmiqp = MPMIQP_Program(A, b, c, H, Q, A_t, b_t, F, binary_indices=[1, 2])
    return mpmiqp


@pytest.fixture()
def mpMILP_market_problem():
    """Simple mpMILP for Seatle-to-Topeka"""

    A = numpy.array(
        [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [-1, 0, -1, 0, 0], [0, -1, 0, -1, -500], [-1, 0, 0, 0, 0], [0, -1, 0, 0, 0],
         [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, -1], [0, 0, 0, 0, 1]], float)
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0, 0, 1], float).reshape(-1, 1)
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], float)
    A_t = numpy.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], float)
    b_t = numpy.array([[1000.0], [1000.0], [0.0], [0.0]], float)
    H = numpy.zeros([5, 2])
    c = numpy.array([178, 187, 187, 151, 50000]).reshape(-1, 1)
    binary_indices = [4]

    milp_prog = MPMILP_Program(A, b, c, H, A_t, b_t, F, binary_indices)
    return milp_prog

@pytest.fixture()
def mpMIQP_market_problem():
    """Simple mpMIQP for Seatle-to-Topeka"""

    A = numpy.array(
        [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [-1, 0, -1, 0, 0], [0, -1, 0, -1, -500], [-1, 0, 0, 0, 0], [0, -1, 0, 0, 0],
         [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, -1], [0, 0, 0, 0, 1]], float)
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0, 0, 1], float).reshape(-1, 1)
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], float)
    A_t = numpy.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], float)
    b_t = numpy.array([[1000.0], [1000.0], [0.0], [0.0]], float)
    H = numpy.zeros([5, 2])
    binary_indices = [4]

    Q = numpy.diag([153, 162, 162, 126, 1])
    c = numpy.array([25, 25, 25, 25, 7.6e6]).reshape(-1, 1)

    miqp_prog = MPMIQP_Program(A, b, c, H, Q, A_t, b_t, F, binary_indices)
    return miqp_prog

@pytest.fixture()
def portfolio_problem_analog():

    num_assets = 8
    S = numpy.diag([i + 1 for i in range(num_assets)])
    mu = [0.09551451, 0.00317183, 0.06799116, 0.12334409, 0.10235298, 0.0754139, 0.00730871, 0.11324299]

    A = numpy.block([[1 for _ in range(num_assets)], [mu[i] for i in range(num_assets)], [-numpy.eye(num_assets)]])
    b = numpy.array([1, 0, *[0 for _ in range(num_assets)]]).reshape(-1, 1)
    F = numpy.block([[0], [1], [numpy.zeros((num_assets, 1))]])
    A_t = numpy.array([[-1], [1]])
    b_t = numpy.array([[-min(mu)], [max(mu)]])
    Q = S
    c = numpy.zeros((num_assets, 1))
    H = numpy.zeros((A.shape[1], F.shape[1]))
    program = MPQP_Program(A, b, c, H, Q, A_t, b_t, F, equality_indices=[0, 1], post_process=False)

    program.solver.solvers['lp'] = 'glpk'
    program.solver.solvers['qp'] = 'quadprog'

    return program

@pytest.fixture()
def bard_mpMILP_adapted_degenerate():
    """
    This is adapted from the inner problem of the bilevel problem given on p. 246 of the book "Practical Bilevel Optimization" by J.F. Bard.
    The upper level variable x is treated as a continuous parameter here.
    Since at the time of implementation of this, PPOPT can't handle a pure mpILP, we add a dummy continuous variable z to the problem.
    z is designed to not affect the objective, but as a result, the problem becomes degenerate.
    """

    program_model = MPModeler()

    x = program_model.add_param(name='x')
    y1 = program_model.add_var(name='y1', vtype=VariableType.binary)
    y2 = program_model.add_var(name='y2', vtype=VariableType.binary)
    y3 = program_model.add_var(name='y3', vtype=VariableType.binary)
    z = program_model.add_var()

    y = y1 + 2*y2 + 4*y3

    program_model.add_constr(x >= 0)
    program_model.add_constr(x <= 10)

    program_model.add_constr(y <= 4)

    program_model.add_constr(-25*x+20*y <= 30)
    program_model.add_constr(x+2*y <= 10)
    program_model.add_constr(2*x-y <= 15)
    program_model.add_constr(2*x+10*y >= 15 + z)

    program_model.add_constr(z >= 0)
    program_model.add_constr(z <= 0.1)

    program_model.set_objective(y)

    program = program_model.formulate_problem()

    return program

@pytest.fixture()
def bard_mpMILP_adapted():
    """
    This is adapted from the inner problem of the bilevel problem given on p. 246 of the book "Practical Bilevel Optimization" by J.F. Bard.
    The upper level variable x is treated as a parameter here.
    Since at the time of implementation of this, PPOPT can't handle a pure mpILP, we add a dummy continuous variable z to the problem.
    By adding z to the objective, we make the problem non-degenerate.
    """


    program_model = MPModeler()

    x = program_model.add_param(name='x')
    y1 = program_model.add_var(name='y1', vtype=VariableType.binary)
    y2 = program_model.add_var(name='y2', vtype=VariableType.binary)
    y3 = program_model.add_var(name='y3', vtype=VariableType.binary)
    z = program_model.add_var()

    y = y1 + 2*y2 + 4*y3

    program_model.add_constr(x >= 0)
    program_model.add_constr(x <= 10)

    program_model.add_constr(y <= 4)

    program_model.add_constr(-25*x+20*y <= 30)
    program_model.add_constr(x+2*y <= 10)
    program_model.add_constr(2*x-y <= 15)
    program_model.add_constr(2*x+10*y >= 15 + z)

    program_model.add_constr(z >= 0)
    program_model.add_constr(z <= 0.1)

    program_model.set_objective(y + z)

    program = program_model.formulate_problem()

    return program

@pytest.fixture()
def bard_mpMILP_adapted_2():
    """
    This is adapted from the inner problem of the bilevel problem given on p. 246 of the book "Practical Bilevel Optimization" by J.F. Bard.
    The upper level variable x is treated as a parameter here.
    Since at the time of implementation of this, PPOPT can't handle a pure mpILP, we add a dummy continuous variable z to the problem.
    The RHS of the constraint x+2y <= 10 is changed to 15 to increase the feasible space, which causes the critical regions to have different properties from the original problem.
    E.g., we can now have fully overlapping regions instead of just partially overlapping regions.
    """


    program_model = MPModeler()

    x = program_model.add_param(name='x')
    y1 = program_model.add_var(name='y1', vtype=VariableType.binary)
    y2 = program_model.add_var(name='y2', vtype=VariableType.binary)
    y3 = program_model.add_var(name='y3', vtype=VariableType.binary)
    z = program_model.add_var()

    y = y1 + 2*y2 + 4*y3

    program_model.add_constr(x >= 0)
    program_model.add_constr(x <= 10)

    program_model.add_constr(y <= 4)

    program_model.add_constr(-25*x+20*y <= 30)
    program_model.add_constr(x+2*y <= 15)
    program_model.add_constr(2*x-y <= 15)
    program_model.add_constr(2*x+10*y >= 15 + z)

    program_model.add_constr(z >= 0)
    program_model.add_constr(z <= 0.1)

    program_model.set_objective(y)

    program = program_model.formulate_problem()

    return program

@pytest.fixture()
def non_negative_least_squares():

    N = 10
    numpy.random.seed(123)
    rng = numpy.random.default_rng(seed=123)
    rev = numpy.random.rand(N)
    rev = (N / numpy.sum(rev)) * rev
    A = random_correlation.rvs(eigs=rev, random_state=rng, tol=10 ** -10)
    y = numpy.random.rand(N)

    m = MPModeler()

    t = m.add_param('lambda')
    x = [m.add_var(name=f'x_[{i}]') for i in range(N)]

    m.add_constrs(x[i] >= 0 for i in range(N))
    m.add_constr(t >= 0)
    m.add_constr(t <= 10)

    z = [sum(A[i, j] * x[j] for j in range(N)) - y[i] for i in range(N)]

    m.set_objective(sum(z[i] ** 2 for i in range(N)) + t * sum(x))

    return m.formulate_problem(process=False)

@pytest.fixture()
def mpMILP_1d():
    m = MPModeler()
    x = m.add_var()
    y = m.add_var(vtype=VariableType.binary)
    t = m.add_param()

    m.add_constr(x >= 0)
    m.add_constr(x + 50 * y >= t)
    m.add_constr(x <= 100)
    m.add_constr(t >= 0)
    m.add_constr(t <= 100)

    m.set_objective(x + 40*y)

    return m.formulate_problem()