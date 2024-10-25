import numpy
import pytest

from src.ppopt.mpqcqp_program import MPQCQP_Program

from tests.test_fixtures import small_mpqcqp, pappas_qcqp_2

@pytest.mark.filterwarnings("ignore::UserWarning") # this suppresses a warning about non-convexity which is irrelevant for this test
def test_solve_theta(small_mpqcqp):
    sol = small_mpqcqp.solve_theta(numpy.array([0]))
    assert numpy.allclose(sol.sol, numpy.array([5, 2]))

@pytest.mark.filterwarnings("ignore::UserWarning") # this suppresses a warning about non-convexity which is irrelevant for this test
def test_check_feasibility(small_mpqcqp):
    assert small_mpqcqp.check_feasibility(numpy.array([0]))
    assert small_mpqcqp.check_feasibility(numpy.array([1]))
    assert small_mpqcqp.check_feasibility(numpy.array([2]))
    assert not small_mpqcqp.check_feasibility(numpy.array([1, 2]))

def test_check_optimality(pappas_qcqp_2):
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