from src.ppopt.problem_generator import generate_mplp, generate_mpqp


def test_mplp_problem_generator():
    # check that this won't give out infeasible problems
    assert generate_mplp(2, 2, 40).feasible_theta_point() is not None


def test_mpqp_problem_generator():
    # check that this won't give out infeasible problems
    assert generate_mpqp(2, 2, 40).feasible_theta_point() is not None
