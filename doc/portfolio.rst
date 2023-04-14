Markowitz Portfolio Allocation
==============================

A fairly common trend in finance has been leveraging mathematical modeling and optimization to give better services to customers and drive better returns. For example, one of the most common decisions of someone who is investing in the stock market (or other markets) is how much of what thing to buy. One of the most common approaches utilizes the Markowitz Portfolio Theory (MPT) concept. Given some assumptions, we can assemble an optimization problem to solve that will give us what asset allocation in the portfolio should be. This concept was developed by the Economist Harry Markowitz, for which he was given the Nobel prize in Economics in 1990. 

The `Markowitz portfolio optimization <https://en.wikipedia.org/wiki/Modern_portfolio_theory>`_ problem is given as follows, where :math:`\Sigma` is the covariance of the commodities. In this example, we are considering only positive definite covariance matrices. :math:`R^*` is the return we wish to get, and :math:`\mu` is the expected rate of return. The constraints effectively specify a return rate that we want, saying that the composition of our portfolio must sum to one (we only have one portfolio!) and that we cannot have negative positions in any commodity.

.. math::
    \begin{align}
    \min_w \quad \frac{1}{2} w^T & \Sigma w\\
    \sum \mu_i w_i &=R^* \\
    \sum w_i &= 1\\
    w_i &\geq 0, \forall i
    \end{align}

This can be reformulated into an parametric quadratic program (pQP), or to say we can solve this for all possible realizations of our desired return and recover the Pareto front of optimal portfolios by switching out :math:`R^*` with an uncertain parameter :math:`\theta`. Since there is only one uncertain parameter, the geometric algorithm is the most efficient for this problem, and even portfolios with hundreds of commodities can be solved in seconds. However since, trying to plot the optimal portfolio positions for hundreds of commodities can be visually busy, we instead are going to solve it for ten commodities so we can still see how the asset allocations change when risk is varried.

Here the covariance matrix and the return coefficients were generated from random numbers. We followed `YALMIP's <https://yalmip.github.io/example/portfolio/>`_ suggestion on generating reasonable enough data. The code in python for how we generate the covariance matrix and the return can be seen below.

.. code-block:: python

    numpy.random.seed(123456789)
    num_assets = 10
    S = numpy.random.randn(num_assets,num_assets)
    S = S@S.T / 10
    mu = numpy.random.rand(num_assets)/100

Here is the problem that we are going to be tackling in this post. Some of the constraints have been noticeably modified. This is due to a standard preprocessing pass that ``ppopt`` runs. This modification increases numerical stability for ill-conditioned optimization problems but has nearly for the problem we are looking at in this example, as it is numerically well conditioned. 

.. code-block:: python

    import numpy
    from ppopt.mpqp_program import MPQP_Program

    A = numpy.block([[1 for i in range(num_assets)],[mu[i] for i in range(num_assets)],[-numpy.eye(num_assets)]])
    b = numpy.array([1,0,*[0 for i in range(num_assets)]]).reshape(-1,1)
    F = numpy.block([[0],[1],[numpy.zeros((num_assets,1))]])
    A_t = numpy.array([[-1],[1]])
    b_t = numpy.array([[-min(mu)],[max(mu)]])
    Q = S
    c = numpy.zeros((num_assets,1))
    H = numpy.zeros((A.shape[1],F.shape[1]))
    portfolio = MPQP_Program(A, b, c, H, Q, A_t, b_t, F,equality_indices= [0,1])

This formulates the parametric problem as follows, we want to parameterize the return :math:`R^*` as :math:`\theta`, so that we can solve over all feasible bounds of return.

.. math::
    \begin{align}
    \min_w \quad \frac{1}{2} w^T & \Sigma w\\
    \sum \mu_i w_i &=\theta \\
    \sum w_i &= 1\\
    w_i &\geq 0, \forall i\\
    \min(\mu) \leq &\theta \leq \max(\mu)
    \end{align}


Now that we have formulated the pQP, all we have to do is solve it. Which can be accomplished with the following python code. We are using the geometric algorithm here, as it is very fast in this type of problem. For this problem it only took half a second to solve.

.. code-block:: python

    from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm

    sol = solve_mpqp(portfolio, mpqp_algorithm.geometric)

To plot the parametric solution of commodities that we should invest in as a function of return, we can just use the inbuilt plotting functionality.

.. code-block:: python

    from ppopt.plot import parametric_plot_1D

    parametric_plot_1D(sol)

.. image:: port_soln.svg

That is fine an good an all, but typically we want to view how this effects the balance of risk and reward. Here we can see the classical shape of the risk-reward tradeoff. The pareto front of all portfolios is completely recovered and is algebraic form.

.. code-block:: python

    import matplotlib.pyplot as plt

    returns = numpy.linspace(min(mu)+ .00001,max(mu) - .000001,1000)
    risk = numpy.array([sol.evaluate_objective(numpy.array([[x]])) for x in returns]).flatten()

    plt.title('Optimal risk v. return pareto front')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.plot(risk,returns)

.. image:: risk_return_port.svg


