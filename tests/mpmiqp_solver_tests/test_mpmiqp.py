
# simple test, just for coverage
import numpy

from src.ppopt.mp_solvers.solve_mpmiqp import solve_mpmiqp
from src.ppopt.mp_solvers.solve_mpqp import mpqp_algorithm
from src.ppopt.mpmilp_program import MPMILP_Program
from tests.test_fixtures import simple_mpMILP, simple_mpMIQP, mpMILP_market_problem, mpMIQP_market_problem, bard_mpMILP_adapted, bard_mpMILP_adapted_2, bard_mpMILP_adapted_degenerate


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

def test_mpmilqp_enumeration_solve_2(mpMIQP_market_problem):

    sol = solve_mpmiqp(mpMIQP_market_problem, cont_algo=mpqp_algorithm.combinatorial)

def test_mpmilp_evaluate(mpMILP_market_problem):

    # find the explicit solution to the mpMILP market problem
    sol = solve_mpmiqp(mpMILP_market_problem)

    # simple test that we are not finding a hole in the middle of two regions
    theta_point = numpy.array([[0.0], [500.0]])

    # get the solution
    ppopt_solution = sol.evaluate(theta_point).flatten()
    ppopt_value = sol.evaluate_objective(theta_point)

    # get the deterministic solution
    det_solution = mpMILP_market_problem.solve_theta(theta_point)
    det_primal_sol = numpy.array(det_solution.sol)

    assert(numpy.isclose(det_solution.obj, ppopt_value))
    assert(all(numpy.isclose(det_primal_sol, ppopt_solution.flatten())))


def test_mpmiqp_evaluate(simple_mpMIQP):

    sol = solve_mpmiqp(simple_mpMIQP)

    sol.evaluate(numpy.array([[1.2]]))


def test_mpmilp_incorrect_algo(simple_mpMILP):

    try:
        sol = solve_mpmiqp(simple_mpMILP, "enum")
        assert(False)
    except TypeError as e:
        print(e)
        assert(True)

def test_mpmilp_cr_removal_1D(bard_mpMILP_adapted):
    sol = solve_mpmiqp(bard_mpMILP_adapted)
    assert(len(sol) == 2)
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[2.]])), 2))
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[3.]])), 1))

def test_mpmilp_cr_removal_1D(bard_mpMILP_adapted_degenerate):
    sol = solve_mpmiqp(bard_mpMILP_adapted_degenerate)
    assert(len(sol) == 5)
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[2.]])), 2))
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[3.]])), 1))

def test_mpmilp_cr_removal_1D_2(bard_mpMILP_adapted_2):
    sol = solve_mpmiqp(bard_mpMILP_adapted_2)
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[2.]])), 2))
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[3.]])), 1))
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[8.]])), 1))
    assert(numpy.isclose(sol.evaluate_objective(numpy.array([[9.]])), 3))