import numpy

from src.ppopt.utils.chebyshev_ball import chebyshev_ball
from src.ppopt.mp_solvers import mpqp_parrallel_combinatorial
from src.ppopt.mp_solvers.solve_mpqp import mpqp_algorithm, solve_mpqp
from tests.test_fixtures import qp_problem, simple_mpLP, portfolio_problem_analog, non_negative_least_squares


def test_solve_mpqp_combinatorial(qp_problem):
    solution = solve_mpqp(qp_problem, mpqp_algorithm.combinatorial)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_gupta_parallel_exp(qp_problem):
    # solution = solve_mpqp(qp_problem, mpqp_algorithm.combinatorial_parallel_exp)

    solution = mpqp_parrallel_combinatorial.solve(qp_problem, 4)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_geometric(qp_problem):
    solution = solve_mpqp(qp_problem, mpqp_algorithm.geometric)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_geometric_parallel(qp_problem):
    solution = solve_mpqp(qp_problem, mpqp_algorithm.geometric_parallel)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_geometric_parallel_exp(qp_problem):
    solution = solve_mpqp(qp_problem, mpqp_algorithm.geometric_parallel_exp)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_graph(qp_problem):
    solution = solve_mpqp(qp_problem, mpqp_algorithm.graph)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_graph_exp(qp_problem):
    solution = solve_mpqp(qp_problem, mpqp_algorithm.graph_exp)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_graph_parallel(qp_problem):
    solution = solve_mpqp(qp_problem, mpqp_algorithm.graph_parallel)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_graph_parallel_exp(qp_problem):
    qp_problem.solver.solvers['lp'] = 'glpk'
    solution = solve_mpqp(qp_problem, mpqp_algorithm.graph_parallel_exp)
    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mpqp_combinatorial_graph(qp_problem):
    qp_problem.solver.solvers['lp'] = 'glpk'
    solution = solve_mpqp(qp_problem, mpqp_algorithm.combinatorial_graph)

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mplp_combinatorial(simple_mpLP):
    solution = solve_mpqp(simple_mpLP, mpqp_algorithm.combinatorial)
    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_missing_algorithm(qp_problem):
    try:
        solution = solve_mpqp(qp_problem, algorithm="cambinatorial")
        assert False
    except TypeError as e:
        print(e)
        assert True


def test_solve_geometric_portfolio(portfolio_problem_analog):
    sol_geo = solve_mpqp(portfolio_problem_analog, mpqp_algorithm.geometric)
    sol_combi = solve_mpqp(portfolio_problem_analog, mpqp_algorithm.combinatorial)

    # they should have the same number of critical regions
    assert(len(sol_geo) == len(sol_combi))

    # test the center of each critical region
    for cr in sol_geo.critical_regions:

        chev_sol = chebyshev_ball(cr.E, cr.f, deterministic_solver=portfolio_problem_analog.solver.solvers['lp'])
        center = chev_sol.sol[0].reshape(-1,1)

        geo_ans = sol_geo.evaluate(center)
        combi_ans = sol_combi.evaluate(center)

        if not numpy.allclose(geo_ans, combi_ans):
            assert False

def test_solve_geometric_nnls(non_negative_least_squares):
    sol_geo = solve_mpqp(non_negative_least_squares, mpqp_algorithm.geometric)
    sol_combi = solve_mpqp(non_negative_least_squares, mpqp_algorithm.combinatorial)

    # they should have the same number of critical regions
    assert(len(sol_geo) == len(sol_combi))

    # test the center of each critical region
    for cr in sol_geo.critical_regions:

        chev_sol = chebyshev_ball(cr.E, cr.f, deterministic_solver=non_negative_least_squares.solver.solvers['lp'])
        center = chev_sol.sol[0].reshape(-1,1)

        geo_ans = sol_geo.evaluate(center)
        combi_ans = sol_combi.evaluate(center)

        if not numpy.allclose(geo_ans, combi_ans):
            assert False
