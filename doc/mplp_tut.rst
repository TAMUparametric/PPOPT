Solving a MPLP Program
======================

Here we are going to solve a classic transportation problem with multiparametric uncertainty. We have a set of plants and a set of markets with corresponding supplies and demand, and we want to minimize the transport cost between the plants, ensuring we satisfy all market demand. The multiparametric formulation is fleshed out in more detail in Multiparametric Optimization and Control by Pistikopolous, Diangelakis, and Oberdieck. This is the mpLP version of introductory mpQP problem, and shows how to solve an mpLP problem.

This optimization problem leads to the following multiparametric optimization problem, with θ representing the markets' uncertain demands. Here we are using a linear cost formulation in the objective


.. math::
    \min_{x} \left[\begin{matrix}178.0\\187.0\\187.0\\151.0\end{matrix}\right]^T\left[\begin{matrix}x_0\\x_1\\x_2\\x_3\end{matrix}\right]


.. math::
    \begin{equation*}
    \begin{split}
    \text{s.t. }\left[\begin{matrix}1.0 & 1.0 & 0 & 0\\0 & 0 & 1.0 & 1.0\\-1.0 & 0 & -1.0 & 0\\0 & -1.0 & 0 & -1.0\\-1.0 & 0 & 0 & 0\\0 & -1.0 & 0 & 0\\0 & 0 & -1.0 & 0\\0 & 0 & 0 & -1.0\end{matrix}\right]\left[\begin{matrix}x_0\\x_1\\x_2\\x_3\end{matrix}\right] & \leq\left[\begin{matrix}350.0\\600.0\\0\\0\\0\\0\\0\\0\end{matrix}\right]+\left[\begin{matrix}0 & 0\\0 & 0\\-1.0 & 0\\0 & -1.0\\0 & 0\\0 & 0\\0 & 0\\0 & 0\end{matrix}\right]\left[\begin{matrix}\theta_0\\\theta_1\end{matrix}\right]\\
    \left[\begin{matrix}1.0 & 0\\0 & 1.0\\-1.0 & 0\\0 & -1.0\end{matrix}\right]\left[\begin{matrix}\theta_0\\\theta_1\end{matrix}\right] & \leq\left[\begin{matrix}1e+03\\1e+03\\0\\0\end{matrix}\right]
    \end{split}
    \end{equation*}

Using PPOPT, this is translated as the following python code. (The latex above was generated for me with ``prog.latex()`` if you were wondering if I typed that all out by hand.)

.. code-block:: python

    import numpy
    from ppopt.mpqp_program import MPLP_Program

    A = numpy.array([[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = numpy.array([178, 187, 187, 151]).reshape(-1,1)
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    CRa = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    CRb = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((A.shape[1],F.shape[1]))

    prog = MPLP_Program(A, b, c, H, CRa, CRb, F)

Alternatively, we can use ``MPModeler`` to build the program. This can be a more user-friendly way to build the program, and it is easier to read and understand. It does not require the user to specify the problem data as matrices, but uses an interface that is more similar to a mathematical formulation.

.. code-block:: python

    from itertools import product
    from ppopt.mpmodel import MPModeler

    model = MPModeler()

    # define problem data
    factories = ['se', 'sd']
    markets = ['ch', 'to']
    capacities = {'se': 350, 'sd': 600}
    cost = {('se', 'ch'): 178, ('se', 'to'): 187, ('sd', 'ch'): 187, ('sd', 'to'): 151}

    # make a variable for each factory-market production pair
    x = {(f, m): model.add_var(name=f'x[{f},{m}]') for f, m in product(factories, markets)}

    # make a parameter for each market demand
    d = {m: model.add_param(name=f'd_[{m}]') for m in markets}

    # bounds on the production capacity of each factory
    model.add_constrs(sum(x[f, m] for m in markets) <= capacities[f] for f in factories)

    # demand satisfaction for each market
    model.add_constrs(sum(x[f, m] for f in factories) >= d[m] for m in markets)

    # bounds on the parametric demand
    model.add_constrs(d[m] <= 1000 for m in markets)
    model.add_constrs(d[m] >= 0 for m in markets)

    # non-negativity of the production variables
    model.add_constrs(x[f, m] >= 0 for f, m in product(factories, markets))

    # set the objective to minimize the total cost
    model.set_objective(sum(cost[f, m] * x[f, m] for f, m in product(factories, markets)))

    prog = model.formulate_problem()


But before you go forward and solve this, I would always recommend processing the constraints. Removing all strongly and weakly redundant constraints and rescaling them leads to significant performance increases and robustifying the numerical stability. In PPOPT, processing the constraints is a simple task.

.. code:: python

    prog.process_constraints()

This results in the following (identical) multiparametric optimization problem. In general removing constraints can exponentially reduce the time to solve explicitly.

.. math::
    \min_{x} \left[\begin{matrix}178.0\\187.0\\187.0\\151.0\end{matrix}\right]^T\left[\begin{matrix}x_0\\x_1\\x_2\\x_3\end{matrix}\right]

.. math::
    \begin{equation*}
    \begin{split}
    \text{s.t. }\left[\begin{matrix}0.7071 & 0.7071 & 0 & 0\\0 & 0 & 0.7071 & 0.7071\\-0.5774 & 0 & -0.5774 & 0\\0 & -0.5774 & 0 & -0.5774\\-1.0 & 0 & 0 & 0\\0 & -1.0 & 0 & 0\\0 & 0 & -1.0 & 0\\0 & 0 & 0 & -1.0\end{matrix}\right]\left[\begin{matrix}x_0\\x_1\\x_2\\x_3\end{matrix}\right] & \leq\left[\begin{matrix}247.5\\424.3\\0\\0\\0\\0\\0\\0\end{matrix}\right]+\left[\begin{matrix}0 & 0\\0 & 0\\-0.5774 & 0\\0 & -0.5774\\0 & 0\\0 & 0\\0 & 0\\0 & 0\end{matrix}\right]\left[\begin{matrix}\theta_0\\\theta_1\end{matrix}\right]\\
    \left[\begin{matrix}1.0 & 0\\0 & 1.0\\-1.0 & 0\\0 & -1.0\end{matrix}\right]\left[\begin{matrix}\theta_0\\\theta_1\end{matrix}\right] & \leq\left[\begin{matrix}1e+03\\1e+03\\0\\0\end{matrix}\right]
    \end{split}
    \end{equation*}

That wasn't that bad, and we were able to cut away some constraints that didn't matter in the process! Now we are ready to solve it. We import the solver functionalities and then specify an algorithm to use. Here we are specifying the combinatorial algorithm. Even though we are using the ``solve_mpqp`` function, this is also the main backend to solve mpLPs!

.. code-block:: python

    from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
    solution = solve_mpqp(prog, mpqp_algorithm.combinatorial)


Now we have the solution, we can either export the solution via the micropop module, or we can plot it. Let's plot it here. The extra arguments mean we are saving a picture of the plot and displaying it to the user (you can give a file path, so it saves somewhere that is not the current working directory).

.. code-block:: python

    from ppopt.plot import parametric_plot

    # saves the plot as 'transport.svg' in the current working directory
    parametric_plot(solution, save_path='transport_mplp', save_format='svg', show=True)

.. image:: transport_mplp.svg