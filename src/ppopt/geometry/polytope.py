import numpy


class Polytope:
    """
    This is a basic convex polytope class in n-dimensions. In future releases this will take the place of explicitly passing \\
    around matrix pairs [A, b] and lead to a simplification of the code base.
    """

    def __init__(self, A: numpy.ndarray, b: numpy.ndarray):
        """Initializes a Polytope."""
        self.A = A
        self.b = b

    def __and__(self, other):  # -> Optional[Polytope]:
        """Takes the union of two convex polytopes."""
        if other is None:
            return Polytope(self.A.copy(), self.b.copy())

        assert type(other) is type(self), f"Polytope Intersection not valid for type {type(other)} "

        A_prime = numpy.block([[self.A], [other.A]])
        b_prime = numpy.block([[self.b], [other.b]])
        return A_prime, b_prime
