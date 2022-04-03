from dataclasses import dataclass
from typing import Optional

import numpy


@dataclass
class SolverOutput:
    """
    This is the generic Solver information object. This will be the general return object from all the back end
    solvers. This was done to remove the need for the user to specialize IO for any particular Solver. It contains
    all the information you would need for the optimization solution including, optimal value, optimal solution,
    the active set, the value of the slack variables and the largange multipliers associated with every constraint (
    these are listed) as the dual variables.

    Members:
    obj: objective value of the optimal solution \n
    sol: x*, numpy array \n

    Optional Parameters -> None or numpy.ndarray type

    slack: the slacks associated with every constraint \n
    equality_indices: the active set of the solution, including strongly and weakly active constraints \n
    dual: the lagrange multipliers associated with the problem\n

    """
    obj: float
    sol: numpy.ndarray

    slack: Optional[numpy.ndarray]
    active_set: Optional[numpy.ndarray]
    dual: Optional[numpy.ndarray]

    def __eq__(self, other):
        if not isinstance(other, SolverOutput):
            return NotImplemented

        return numpy.allclose(self.slack, other.slack) and numpy.allclose(self.active_set,
                                                                          other.active_set) and numpy.allclose(
            self.dual, other.dual) and numpy.allclose(self.sol, other.sol) and numpy.allclose(self.obj, other.obj)


def get_program_parameters(Q: Optional[numpy.ndarray], c: Optional[numpy.ndarray], A: Optional[numpy.ndarray],
                           b: Optional[numpy.ndarray]):
    """ Given a set of possibly None optimization parameters determine the number of variables and constraints """
    num_c = 0
    num_v = 0

    if Q is not None:
        num_v = Q.shape[0]

    if A is not None:
        num_v = A.shape[1]
        num_c = A.shape[0]

    if c is not None:
        num_v = numpy.size(c)

    return num_v, num_c
