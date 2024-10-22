import numpy

from src.ppopt.mpqcqp_program import MPQCQP_Program

from tests.test_fixtures import small_mpqcqp

def test_solve_theta(small_mpqcqp):
    sol = small_mpqcqp.solve_theta(numpy.array([0]))
    assert numpy.allclose(sol.sol, numpy.array([5, 2]))

def test_check_feasibility(small_mpqcqp):
    assert small_mpqcqp.check_feasibility(numpy.array([0]))
    assert small_mpqcqp.check_feasibility(numpy.array([1]))
    assert small_mpqcqp.check_feasibility(numpy.array([2]))
    assert not small_mpqcqp.check_feasibility(numpy.array([1, 2]))