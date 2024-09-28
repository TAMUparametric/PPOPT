Maximal Controller Gain 
=======================

In some optimal control applications it can be beneficial to not just solve the optimal control problem (in this case a Model Predctive Control (MPC) problem), but also have a bound on the controller gain to verify rubustness claims of the controller. In this example we will show how to compute the maximum gain of a MPC using PPOPT. Here we will use the an example 10.4.1 from the book 'Multi-Parametric Optimization and Control' by Pistikopoulos, Diangelakis, and Oberdieck.


.. math::

    \begin{align}
        \min_{u, x} \quad & x_{10}^T\begin{bmatrix}2.6235 & 1.6296\\ 1.6296 & 2.6457\end{bmatrix}x_{10} + \sum_{k=0}^9\left(x_k^Tx_k + 0.01u_k^2\right)\\
        \text{s.t. }x_{k+1} &= \begin{bmatrix}1 & 1 \\ 0 & 1\end{bmatrix}x_k + \begin{bmatrix}0 \\ 1\end{bmatrix}u_k, \forall k \in [0,9]\\
        &-25\leq \begin{bmatrix}1 & 2 \end{bmatrix}x_k \leq 25, \forall k \in [0,10]\\
        &-1 \leq u_k \leq 1, \forall k \in [0,9]\\
        &\begin{bmatrix}
            -10 \\ -10
        \end{bmatrix} \leq u_k \leq \begin{bmatrix}
            10 \\ 10
        \end{bmatrix}, \forall k \in [0,10]\\
        &\begin{bmatrix}
            0.6136 & 1.6099\\
            -.3742 & -0.3682\\
            -0.6136 & -1.6099\\
            .3742 & 0.3682\\
        \end{bmatrix}x_{10} \leq \begin{bmatrix}
            1 \\ 1 \\ 1 \\ 1
        \end{bmatrix}
    \end{align}

.. code:: python

    num_x = 2
    N = 10

    m = MPModeler()


    # stabalizing terminal weight
    P = numpy.array([[2.6235, 1.6296],[1.6296, 2.6457]])

    # add state and input variables to the model
    u = [m.add_var(f'u_[{N}]') for t in range(N-1)]
    x = [[m.add_var(f'x_[{N},{i}]') for i in range(num_x)] for t in range(N)]

    # add initital state params to model
    x_0 = [m.add_param(f'x_0[{i}]') for i in range(num_x)]

    # give bounds to the input actions
    m.add_constrs(u[k] <= 1 for k in range(N-1))
    m.add_constrs(u[k] >= -1 for k in range(N-1))

    # give bounds to the states
    m.add_constrs(x[k][nx] <= 10 for k in range(N) for nx in range(num_x))
    m.add_constrs(x[k][nx] >= -10 for k in range(N) for nx in range(num_x))

    # operational bounds
    m.add_constrs(x[k][0] + 2*x[k][1] <= 25 for k in range(N))
    m.add_constrs(x[k][0] + 2*x[k][1] >= -25 for k in range(N))

    # enforce initial state
    m.add_constrs(x_0[nx] == x[0][nx] for nx in range(num_x))

    # enforce the state space model
    m.add_constrs(x[k+1][0] == x[k][0] + x[k][1] for k in range(N-1))
    m.add_constrs(x[k+1][1] == x[k][1] + u[k] for k in range(N-1))

    # terminal constraint set
    m.add_constr(0.6136*x[-1][0] + 1.6099*x[-1][1] <= 1)
    m.add_constr(-0.3742*x[-1][0] - 0.3682*x[-1][1] <= 1)
    m.add_constr(-0.6136*x[-1][0] - 1.6099*x[-1][1] <= 1)
    m.add_constr(0.3742*x[-1][0] + 0.3682*x[-1][1] <= 1)


    # declare the objective
    terminal_objective = sum(P[i,j]*x[-1][i]*x[-1][j] for i in range(num_x) for j in range(num_x))
    m.set_objective(sum(x_t[0]**2 + x_t[1]**2 for x_t in x) + 0.01*sum(u_t**2 for u_t in u) + terminal_objective)


    prog = m.formulate_problem()

Now that the problem is formulated, we can solve it, we are going to be using the graph algorithm to solve this problem

.. code:: python

    from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm

    sol = solve_mpqp(prog, mpqp_algorithm.graph_parallel)

