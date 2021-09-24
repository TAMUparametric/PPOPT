Multiparametric Algorithms
==========================
This section will give a basic overview of the algorithms used in this package for solving the multiparametric programming problems. All current algorithms depend on exploring the theta space of the problem, choosing differing ways to do so.


Geometric Algorithm
-------------------
The Geometric algorithm is based on exporing the feasible space. The name of the algorithm comes from the fact that it is geometrically exploring the space by flipping critical region facets.

This is best used in situations where the number of theta dimensions is small and scales well with number of variables and constraints.

Combinatorial Algorithm
-----------------------

The combinatorial algorithm is based on exploring feasible active set combinations. It is called the combinatorial algorithm due the fact that th.... This is the most robust multiparametric algorithm as it handles both primal and dual degeneracy and will always fully solve the multiparametric programming problem.

This is best used in situations where the number of constraints and variables is small and scales well with number of theta dimensions.

Graph Algorithm
---------------
The graph algorithm is based on exploring the connected graph of active set combinations. This can be viewed as a combination certain aspects of the Geometric algorithm and the Combinatorial algorithm.

This is a general purpose algorithm that scales ok with number of variables, constraints, and parameters. However this algorithm has been shown to be unstable in that it fails to fully solve some multiparametric programs with poor conditioning.
