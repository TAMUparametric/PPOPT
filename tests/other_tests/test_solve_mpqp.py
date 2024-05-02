from src.ppopt.mp_solvers import mpqp_parrallel_combinatorial
from src.ppopt.mp_solvers.solve_mpqp import mpqp_algorithm, solve_mpqp
from src.ppopt.plot import parametric_plot
from tests.test_fixtures import qp_problem, simple_mpLP


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
    solution = solve_mpqp(qp_problem, mpqp_algorithm.combinatorial_graph)\

    assert solution is not None
    assert len(solution.critical_regions) == 4


def test_solve_mplp_combinatorial(simple_mpLP):
    solution = solve_mpqp(simple_mpLP, mpqp_algorithm.combinatorial)
    assert solution is not None
    assert len(solution.critical_regions) == 4


