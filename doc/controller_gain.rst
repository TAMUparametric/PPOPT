Maximal Controller Gain 
=======================

In some optimal control applications it can be beneficial to not just solve the optimal control problem, but also have a bound on the controller gain to verify robustness of a controller. In this example we will show how to compute the maximum gain of an MPC using PPOPT. Here we will use the an example 10.4.1 from the book 'Multi-Parametric Optimization and Control' by Pistikopoulos, Diangelakis, and Oberdieck.

The maximum gain of a controller can be seen as the :math:`\kappa_p`, where :math:`\theta_1` and :math:`\theta_0` can be seen as realizations different process states, and :math:`u_0(\theta_1)` and :math:`u_0(\theta_0)` are the corresponding initial control actions that are applied at the first time stage of each MPC.

.. math::
    || u_0(\theta_1) - u_0(\theta_0)||_p \leq \kappa_p ||\theta_1 - \theta_0||_p

The mathematical description of the controller that we are looking at today can be seen below. This is based on a state space model, with constraints on state, inputs, operational constraints, and a terminal set constraint. This considers 10 time steps into the future.

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

This multiparametric program can be modeled in PPOPT, with the ``MPModeler`` interface.

.. code:: python

    import numpy
    from ppopt.mpmodel import MPModeler

    num_x = 2
    N = 10

    m = MPModeler()


    # stabalizing terminal weight
    P = numpy.array([[2.6235, 1.6296],[1.6296, 2.6457]])

    # add state and input variables to the model
    u = [m.add_var(f'u_[{t}]') for t in range(N-1)]
    x = [[m.add_var(f'x_[{t},{i}]') for i in range(num_x)] for t in range(N)]

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

    # generate the mpp from the symbolic definition
    prog = m.formulate_problem()

Now that the problem is formulated, we can solve it, we are going to be using one of the parallel graph algorithms to solve this problem.

.. code:: python

    from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm

    sol = solve_mpqp(prog, mpqp_algorithm.graph_parallel_exp)


With the explicit solution now in hand, we can evaluate the the gain of the controller. It was shown in 'On the maximal controller gain in linear MPC' by Darun et al, that if we have a explicit solution that is a continuous piecewise affine function which is true for an mpMPC based on mpQP, then we can compute :math:`\kappa_p` in a rather simple way.

.. math::

    \begin{align}
        u_0(\theta) &= \begin{cases}
            A_0\theta + b_0 \text{ if } \theta \in \Theta_0\\
            \dots\\\
            A_J\theta + b_J\text{ if } \theta \in \Theta_J
        \end{cases}\\
        \kappa_p &= \max_{j\in [0,\dots,J]}||K_j||_p
    \end{align}

Implementing this in code, we take the piece of the explicit solution relating to :math:`u_0(\theta)`, which here is just taking the row from the explicit solution relating to the initial input action. We can then compute :math:`\kappa_1`, which is equal to 1.61., directly from the explicit solution. Other :math:`\kappa_p` values can be computed by changing the norm that we are taking in the max function.

.. code:: python

    # get the index of the variable from the modeler
    idx = [_ for idx, v in enumerate(m.variables) if "u_[0]" == v.name]
    kappa_1 = max(numpy.linalg.norm(cr.A[idx],1), for cr in sol.critical_regions)

