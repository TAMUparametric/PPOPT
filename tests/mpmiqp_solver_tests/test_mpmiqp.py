
# simple test, just for coverage
import numpy

from src.ppopt.mpmilp_program import MPMILP_Program
from tests.test_fixtures import simple_mpMILP


def test_mpmilp_process_constraints(simple_mpMILP):
    simple_mpMILP.process_constraints([0,1])


def test_mpmilp_sub_problem(simple_mpMILP):
    sub_problem = simple_mpMILP.generate_substituted_problem([0 ,1])

    print(sub_problem.A)
    print(sub_problem.b)
    print(sub_problem.F)
    print(sub_problem.equality_indices)

    # this should generate the following determanistic problem
    # min -3x_1 s.t. x = 0, x <= theta, |theta| <= 2

