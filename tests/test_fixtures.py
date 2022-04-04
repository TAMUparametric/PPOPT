import pytest
import numpy

from src.ppopt.critical_region import CriticalRegion
from src.ppopt.mplp_program import MPLP_Program
from src.ppopt.mpmilp_program import MPMILP_Program
from src.ppopt.mpmiqp_program import MPMIQP_Program
from src.ppopt.mpqp_program import MPQP_Program
from src.ppopt.mp_solvers.mpqp_combinatorial import CombinationTester
from src.ppopt.mp_solvers.solve_mpqp import solve_mpqp
from src.ppopt.solution import Solution
from src.ppopt.utils.general_utils import make_column


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


@pytest.fixture
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
    return MPLP_Program(A, b, c, H, A_t, b_t, F, None, None, None, equality_indices = [0])


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

    prog = MPQP_Program(A, b, c, H, Q, CRa, CRb, F)
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

    mpmiqp = MPMIQP_Program(A, b, c, H, Q,A_t, b_t, F, binary_indices=[1, 2])
    return mpmiqp

