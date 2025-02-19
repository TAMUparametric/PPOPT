# simple test, just for coverage
import numpy

from src.ppopt.mp_solvers.solve_mpmiqp import solve_mpmiqp
from src.ppopt.mp_solvers.solve_mpqp import mpqp_algorithm, solve_mpqp
from tests.test_fixtures import simple_mpMILP, simple_mpMIQP, mpMILP_market_problem, mpMIQP_market_problem, \
    bard_mpMILP_adapted, bard_mpMILP_adapted_2, bard_mpMILP_adapted_degenerate, mpMILP_1d, acevedo_mpmilp, \
    pappas_multi_objective, pappas_multi_objective_2
from src.ppopt.utils.mpqp_utils import get_bounds_1d


def test_mpmilp_process_constraints(simple_mpMILP):
    simple_mpMILP.process_constraints([0, 1])


def test_mpmiqp_process_constraints(simple_mpMIQP):
    simple_mpMIQP.process_constraints([0, 1])


def test_mpmilp_sub_problem(simple_mpMILP):
    sub_problem = simple_mpMILP.generate_substituted_problem([0, 1])

    assert (sub_problem.A.shape == (2, 1))
    assert (sub_problem.equality_indices == [0])


def test_mpmilp_partial_feasibility(simple_mpMILP):
    assert (simple_mpMILP.check_bin_feasibility([0, 0]))
    assert (simple_mpMILP.check_bin_feasibility([1, 0]))
    assert (simple_mpMILP.check_bin_feasibility([0, 1]))
    assert (not simple_mpMILP.check_bin_feasibility([1, 1]))

    # this should generate the following determanistic problem
    # min -3x_1 s.t. x = 0, x <= theta, |theta| <= 2


def test_mpmiqp_partial_feasibility(simple_mpMIQP):
    assert (simple_mpMIQP.check_bin_feasibility([0, 0]))
    assert (simple_mpMIQP.check_bin_feasibility([1, 0]))
    assert (simple_mpMIQP.check_bin_feasibility([0, 1]))
    assert (not simple_mpMIQP.check_bin_feasibility([1, 1]))

    # this should generate the following determanistic problem
    # min -3x_1 s.t. x = 0, x <= theta, |theta| <= 2


def test_mpmilp_enumeration_solve(simple_mpMILP):
    sol = solve_mpmiqp(simple_mpMILP, num_cores=1)


def test_mpmilqp_enumeration_solve(simple_mpMIQP):
    sol = solve_mpmiqp(simple_mpMIQP, num_cores=1)


def test_mpmilqp_enumeration_solve_2(mpMIQP_market_problem):
    sol = solve_mpmiqp(mpMIQP_market_problem, cont_algo=mpqp_algorithm.combinatorial, num_cores=1)


def test_mpmilp_evaluate(mpMILP_market_problem):
    # find the explicit solution to the mpMILP market problem
    sol = solve_mpmiqp(mpMILP_market_problem, num_cores=1)

    # simple test that we are not finding a hole in the middle of two regions
    theta_point = numpy.array([[0.0], [500.0]])

    # get the solution
    ppopt_solution = sol.evaluate(theta_point).flatten()
    ppopt_value = sol.evaluate_objective(theta_point)

    # get the deterministic solution
    det_solution = mpMILP_market_problem.solve_theta(theta_point)
    det_primal_sol = numpy.array(det_solution.sol)

    assert (numpy.isclose(det_solution.obj, ppopt_value))
    assert (all(numpy.isclose(det_primal_sol, ppopt_solution.flatten())))


def test_mpmiqp_evaluate(simple_mpMIQP):
    sol = solve_mpmiqp(simple_mpMIQP, num_cores=1)

    sol.evaluate(numpy.array([[1.2]]))


def test_mpmilp_incorrect_algo(simple_mpMILP):
    try:
        sol = solve_mpmiqp(simple_mpMILP, "enum")
        assert (False)
    except TypeError as e:
        print(e)
        assert (True)


def test_mpmilp_cr_removal_1D(bard_mpMILP_adapted):
    sol = solve_mpmiqp(bard_mpMILP_adapted, num_cores=1)
    assert (len(sol) == 2)
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[2.]])), 2))
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[3.]])), 1))


def test_mpmilp_cr_removal_1D(bard_mpMILP_adapted_degenerate):
    sol = solve_mpmiqp(bard_mpMILP_adapted_degenerate, num_cores=1)
    assert (len(sol) == 5)
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[2.]])), 2))
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[3.]])), 1))


def test_mpmilp_cr_removal_1D_2(bard_mpMILP_adapted_2):
    sol = solve_mpmiqp(bard_mpMILP_adapted_2, num_cores=1)
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[2.]])), 2))
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[3.]])), 1))
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[8.]])), 1))
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[9.]])), 3))


def test_small_mpmilp_1d(mpMILP_1d):
    sol = solve_mpmiqp(mpMILP_1d, num_cores=1)
    assert (len(sol) == 3)
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[2.]])), 2))
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[45.]])), 40))
    assert (numpy.isclose(sol.evaluate_objective(numpy.array([[60.]])), 50))

    expected_bounds = [(0, 40), (40, 50), (50, 100)]
    for cr in sol.critical_regions:
        lb, ub = get_bounds_1d(cr.E, cr.f)
        assert any(numpy.isclose(lb, expected_lb) and numpy.isclose(ub, expected_ub) for expected_lb, expected_ub in
                   expected_bounds)


def test_acevedo_mpmilp(acevedo_mpmilp):
    # for small problems the cost of running the parallel pool is higher than the cost of solving the problem serially
    sol = solve_mpmiqp(acevedo_mpmilp, num_cores=1)

    theta_point = numpy.array([[0.5] * 3]).reshape(-1, 1)
    det_sol = acevedo_mpmilp.solve_theta(theta_point)

    assert (numpy.allclose(sol.evaluate(theta_point).flatten(), det_sol.sol))
    assert (numpy.allclose(sol.evaluate_objective(theta_point), det_sol.obj))


def test_pappas_mpmilp(pappas_multi_objective):
    sol = solve_mpmiqp(pappas_multi_objective, num_cores=1)

    assert (len(sol) == 3)


def test_pappas_mpmilp_2(pappas_multi_objective_2):
    sol = solve_mpmiqp(pappas_multi_objective_2, num_cores=1)

    theta_point = numpy.array([[90.0]])

    theta_sol = sol.evaluate(theta_point)
    theta_obj = sol.evaluate_objective(theta_point)

    # deterministic solution at theta = 90
    det_sol = pappas_multi_objective_2.solve_theta(theta_point)

    assert numpy.allclose(theta_sol.flatten(), det_sol.sol)
    assert numpy.allclose(theta_obj, det_sol.obj)
