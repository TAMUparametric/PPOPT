from src.ppopt.utils.mpqp_utils import gen_cr_from_active_set

from tests.test_fixtures import quadratic_program, qp_problem, simple_mpqp_problem

def test_check_feasibility_1(quadratic_program):
    assert quadratic_program.check_feasibility([])


def test_check_feasibility_2(quadratic_program):
    assert quadratic_program.check_feasibility([0])


def test_check_feasibility_3(quadratic_program):
    assert not quadratic_program.check_feasibility([6, 7, 8], False)


def test_check_optimality(simple_mpqp_problem):
    assert simple_mpqp_problem.check_optimality([]) is not None
    assert simple_mpqp_problem.check_optimality([0]) is None
    assert simple_mpqp_problem.check_optimality([1]) is not None
    assert simple_mpqp_problem.check_optimality([0, 1]) is None


def test_build_cr_from_active_set(qp_problem):
    qp_problem.scale_constraints()

    assert gen_cr_from_active_set(qp_problem, [2, 3]) is not None
    assert gen_cr_from_active_set(qp_problem, [0, 2, 3]) is not None
    assert gen_cr_from_active_set(qp_problem, [0, 2, 3, 4]) is not None
    assert gen_cr_from_active_set(qp_problem, [0, 2, 3, 4]) is not None
