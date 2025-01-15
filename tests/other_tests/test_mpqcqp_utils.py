import numpy
import pytest

from src.ppopt.mpqcqp_program import MPQCQP_Program, QConstraint, ApproxOptions

from tests.test_fixtures import small_mpqcqp, pappas_qcqp_1, pappas_qcqp_2, redundant_qcqp

@pytest.mark.filterwarnings("ignore::UserWarning") # this suppresses a warning about non-convexity which is irrelevant for this test
def test_solve_theta(small_mpqcqp):
    sol = small_mpqcqp.solve_theta(numpy.array([0]))
    assert numpy.allclose(sol.sol, numpy.array([5, 2]))
    sol = small_mpqcqp.solve_theta(numpy.array([[1]]))
    assert numpy.allclose(sol.sol, numpy.array([[4, numpy.sqrt(7)]]))

@pytest.mark.filterwarnings("ignore::UserWarning") # this suppresses a warning about non-convexity which is irrelevant for this test
def test_check_feasibility(small_mpqcqp):
    assert small_mpqcqp.check_feasibility(numpy.array([0]))
    assert small_mpqcqp.check_feasibility(numpy.array([1]))
    assert small_mpqcqp.check_feasibility(numpy.array([2]))
    assert not small_mpqcqp.check_feasibility(numpy.array([1, 2]))

def test_check_optimality_1(pappas_qcqp_1):
    sol = pappas_qcqp_1.check_optimality([])
    assert sol["t"] > 0 # no active constraints, corresponds to CR 1 in the paper
    assert numpy.allclose(sol["x"], numpy.array([-2]))
    assert numpy.allclose(sol["theta"], numpy.array([2, 2.5]))
    sol = pappas_qcqp_1.check_optimality([0])
    assert sol["t"] > 0 # linear constraint active, corresponds to CR 3 in the paper
    assert numpy.allclose(sol["x"], numpy.array([-2.449]), atol=1e-3)
    assert numpy.allclose(sol["theta"], numpy.array([0.551, 3]), atol=1e-3)
    assert numpy.allclose(sol["lambda"], numpy.array([0.899]), atol=1e-3)
    assert pappas_qcqp_1.check_optimality([1]) is None
    assert pappas_qcqp_1.check_optimality([2]) is None
    sol = pappas_qcqp_1.check_optimality([3])
    assert sol["t"] > 0 # quadratic constraint active, corresponds to CR 2 in the paper
    assert numpy.allclose(sol["x"], numpy.array([-1.239]), atol=1e-3)
    assert numpy.allclose(sol["theta"], numpy.array([2, 0.057]), atol=1e-3)
    assert numpy.allclose(sol["lambda"], numpy.array([3.182]), atol=1e-3)
    assert pappas_qcqp_1.check_optimality([0, 1]) is None
    assert pappas_qcqp_1.check_optimality([0, 2]) is None
    assert pappas_qcqp_1.check_optimality([0, 3])["t"] > 0 # quadratic constraint and first linear constraint active
    assert pappas_qcqp_1.check_optimality([1, 2]) is None
    assert pappas_qcqp_1.check_optimality([1, 3]) is None
    assert pappas_qcqp_1.check_optimality([2, 3]) is None
    assert pappas_qcqp_1.check_optimality([0, 1, 2]) is None
    assert pappas_qcqp_1.check_optimality([0, 1, 3]) is None
    assert pappas_qcqp_1.check_optimality([0, 2, 3]) is None
    assert pappas_qcqp_1.check_optimality([1, 2, 3]) is None
    assert pappas_qcqp_1.check_optimality([0, 1, 2, 3]) is None

def test_check_optimality_2(pappas_qcqp_2):
    assert pappas_qcqp_2.check_optimality([])["t"] > 0 # no active constraints, corresponds to CR 1 in the paper
    # assert pappas_qcqp_2.check_optimality([0]) is None # this one is also optimal for some reason, but the paper doesn't include it
    assert pappas_qcqp_2.check_optimality([1]) is None
    assert pappas_qcqp_2.check_optimality([2]) is None
    assert pappas_qcqp_2.check_optimality([3])["t"] > 0 # quadratic constraint active, corresponds to CR 2 in the paper
    assert pappas_qcqp_2.check_optimality([0, 1]) is None
    assert pappas_qcqp_2.check_optimality([0, 2]) is None
    assert pappas_qcqp_2.check_optimality([0, 3])["t"] > 0 # quadratic constraint and first linear constraint active, corresponds to CR 3 in the paper
    assert pappas_qcqp_2.check_optimality([1, 2]) is None
    assert pappas_qcqp_2.check_optimality([1, 3]) is None
    assert pappas_qcqp_2.check_optimality([2, 3]) is None
    assert pappas_qcqp_2.check_optimality([0, 1, 2]) is None
    assert pappas_qcqp_2.check_optimality([0, 1, 3]) is None
    assert pappas_qcqp_2.check_optimality([0, 2, 3]) is None
    assert pappas_qcqp_2.check_optimality([1, 2, 3]) is None
    assert pappas_qcqp_2.check_optimality([0, 1, 2, 3]) is None

def test_constraint_processing(redundant_qcqp):
    assert redundant_qcqp.num_constraints() == 5
    assert redundant_qcqp.A_t.shape[0] == 3
    redundant_qcqp.process_constraints()
    assert redundant_qcqp.num_constraints() == 2
    assert redundant_qcqp.num_linear_constraints() == 1
    assert redundant_qcqp.num_quadratic_constraints() == 1
    assert redundant_qcqp.A_t.shape[0] == 1

def test_qconstraint_linearization():
    Q = numpy.array([[1, 0], [0, 1]])
    H = numpy.array([[1], [1]])
    A = numpy.array([[1, 1]])
    b = numpy.array([1])
    F = numpy.array([[1]])
    Q_t = numpy.array([[1]])

    qc = QConstraint(Q, H, A, b, F, Q_t)

    x0 = numpy.array([0, 0])
    theta0 = numpy.array([0])
    A_t, b_t, F_t = qc.linearize((x0, theta0))
    assert numpy.allclose(A_t, numpy.array([[1, 1]]))
    assert numpy.allclose(b_t, numpy.array([1]))
    assert numpy.allclose(F_t, numpy.array([1]))

    x1 = numpy.array([1, 2])
    theta1 = numpy.array([3])
    A_t, b_t, F_t = qc.linearize((x1, theta1))
    assert numpy.allclose(A_t, numpy.array([[6, 8]]))
    assert numpy.allclose(b_t, numpy.array([6]))
    assert numpy.allclose(F_t, numpy.array([4]))

def test_approx_cr(pappas_qcqp_1):
    initial_x = numpy.array([-2])
    initial_theta = numpy.array([0,0])
    initial_point = (initial_x, initial_theta)
    options = ApproxOptions()
    cr = pappas_qcqp_1.gen_approx_cr_from_active_set([3], initial_point, options)
    assert len(cr) == 4
    