
# simple test, just for coverage
import numpy

from src.ppopt.mpmilp_program import MPMILP_Program
from src.ppopt.mp_solvers.solve_mpmiqp import solve_mpmiqp
from tests.test_fixtures import simple_mpMILP, simple_mpMIQP


def test_mpmilp_process_constraints(simple_mpMILP):
    simple_mpMILP.process_constraints([0,1])

def test_mpmiqp_process_constraints(simple_mpMIQP):
    simple_mpMIQP.process_constraints([0,1])

def test_mpmilp_sub_problem(simple_mpMILP):
    sub_problem = simple_mpMILP.generate_substituted_problem([0 ,1])

    assert(sub_problem.A.shape == (2,1))
    assert(sub_problem.equality_indices == [0])

def test_mpmilp_partial_feasibility(simple_mpMILP):

    assert(simple_mpMILP.check_bin_feasibility([0,0]))
    assert(simple_mpMILP.check_bin_feasibility([1, 0]))
    assert(simple_mpMILP.check_bin_feasibility([0, 1]))
    assert(not simple_mpMILP.check_bin_feasibility([1, 1]))

    # this should generate the following determanistic problem
    # min -3x_1 s.t. x = 0, x <= theta, |theta| <= 2

def test_mpmiqp_partial_feasibility(simple_mpMIQP):

    assert(simple_mpMIQP.check_bin_feasibility([0,0]))
    assert(simple_mpMIQP.check_bin_feasibility([1, 0]))
    assert(simple_mpMIQP.check_bin_feasibility([0, 1]))
    assert(not simple_mpMIQP.check_bin_feasibility([1, 1]))

    # this should generate the following determanistic problem
    # min -3x_1 s.t. x = 0, x <= theta, |theta| <= 2


def test_mpmilp_enumeration_solve(simple_mpMILP):

    sol = solve_mpmiqp(simple_mpMILP)

def test_mpmilqp_enumeration_solve(simple_mpMIQP):

    sol = solve_mpmiqp(simple_mpMIQP)

def test_mpmilp_evaluate(simple_mpMILP):

    sol = solve_mpmiqp(simple_mpMILP)

    sol.evaluate(numpy.array([[1.2]]))

def test_mpmiqp_evaluate(simple_mpMIQP):

    sol = solve_mpmiqp(simple_mpMIQP)

    sol.evaluate(numpy.array([[1.2]]))
