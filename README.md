# PPOPT

[![Python package](https://github.com/TAMUparametric/PPOPT/actions/workflows/PPOPT_CI.yml/badge.svg)](https://github.com/TAMUparametric/PPOPT/actions/workflows/PPOPT_CI.yml)
[![Documentation Status](https://readthedocs.org/projects/ppopt/badge/?version=latest)](https://ppopt.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/ppopt.svg)](https://pypi.org/project/ppopt)
[![Downloads](https://static.pepy.tech/personalized-badge/ppopt?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/ppopt)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a7df65fcf0104c2ab7fd0105f10854c6)](https://app.codacy.com/gh/TAMUparametric/PPOPT/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

**P**ython **P**arametric **OP**timization **T**oolbox (**PPOPT**) is a software platform for solving and manipulating
multiparametric programs in Python. 

## Installation

Currently, PPOPT requires Python 3.8 or higher and can be installed with the following commands.

```bash
pip install ppopt
```

To install PPOPT and install all optional solvers the following installation is recommended. 

```bash 
pip install ppopt[optional]
```

In Python 3.11 and beyond there is currently an error with the quadprog package. An alternate version that fixed this error can be installed here.

```bash
pip install git+https://github.com/HKaras/quadprog/
```

## Completed Features

- Solver interface for mpLPs and mpQPs with the following algorithms
    1. Serial and Parallel Combinatorial Algorithms
    2. Serial and Parallel Geometrical Algorithms
    3. Serial and Parallel Graph based Algorithms
- Solver interface for mpMILPs and mpMIQPs with the following algorithms
    1. Enumeration based algorithm
- Multiparametric solution export to C++, JavaScript, and Python
- Plotting utilities
- Presolver and Conditioning for Multiparametric Programs

## Key Applications

- Explicit Model Predictive Control
- Multilevel Optimization
- Integrated Design, Control, and Scheduling
- Robust Optimization

For more information about Multiparametric programming and it's
applications, [this paper](https://www.frontiersin.org/articles/10.3389/fceng.2020.620168/full) is a good jumping point.

## Quick Overview

To give a fast primer of what we are doing, we are solving multiparametric programming problems (fast) by writing
parallel algorithms efficiently. Here is a quick scaling analysis on a large multiparametric program with the
combinatorial algorithm.

![image](https://github.com/TAMUparametric/PPOPT/blob/main/Figures/loglog_scaling.png?raw=true)
![image](https://github.com/TAMUparametric/PPOPT/blob/main/Figures/scaleing_eff.png?raw=true)

Here is a benchmark against the state of the art multiparametric programming solvers. All tests run on the Terra
Supercomputer at Texas A&M University. Matlab 2021b was used for solvers written in matlab and Python 3.8 was used for
PPOPT.

![image](https://github.com/TAMUparametric/PPOPT/blob/main/Figures/bench.png?raw=true)

## Citation

Since a lot of time and effort has gone into PPOPT's development, please cite the following publication if you are using
PPOPT for your own research.

```text
@incollection{kenefake2022ppopt,
  title={PPOPT-Multiparametric Solver for Explicit MPC},
  author={Kenefake, Dustin and Pistikopoulos, Efstratios N},
  booktitle={Computer Aided Chemical Engineering},
  volume={51},
  pages={1273--1278},
  year={2022},
  publisher={Elsevier}
}
```
