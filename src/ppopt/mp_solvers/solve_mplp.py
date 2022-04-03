from enum import Enum

from ..mplp_program import MPLP_Program


class mplp_solver(Enum):
    Dustin = '1'


def solve_mplp(problem: MPLP_Program, algorithm: mplp_solver = mplp_solver.Dustin):
    """
    This is the main solver interface for MPLP type problems.

    :param problem:
    :param algorithm:
    :return:
    """
    pass
