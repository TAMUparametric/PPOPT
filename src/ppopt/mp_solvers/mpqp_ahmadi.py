from typing import Optional

from ..mpqp_program import MPQP_Program
from ..solution import Solution


def solve(program: MPQP_Program) -> Optional[Solution]:
    """
    This solves a MPQP program with the method proposed by Parisa Ahmadi-Moshkenani et al. This algorithm is similar
    to the graph algorithm proposed by Richard Oberdeik.

    The source for the algorithm can be found here. https://ieeexplore.ieee.org/document/8252719

    :param program: a MPQP_Program
    :return: The solution of a MPQP Program
    """
    pass
