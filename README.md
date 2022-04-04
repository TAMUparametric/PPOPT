# PPOPT

[![Python package](https://github.com/TAMUparametric/PPOPT/actions/workflows/python-package.yml/badge.svg)](https://github.com/TAMUparametric/PPOPT/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/ppopt/badge/?version=latest)](https://ppopt.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/ppopt.svg)](https://pypi.org/project/ppopt)
[![Downloads](https://static.pepy.tech/personalized-badge/ppopt?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/ppopt)

**P**ython **P**arametric **OP**timization **T**oolbox (**PPOPT**) is a software platform for solving and manipulating multiparametric programs in Python. 

## Installation

Currently, PPOPT requires Python 3.7 or higher and can be installed with the following commands.

```bash
pip install -e git+https://github.com/mmihaltz/pysettrie.git#egg=pysettrie
pip install ppopt
```

To install PPOPT and install all optional solvers the following installation is recommended.

```bash 
pip install -e git+https://github.com/mmihaltz/pysettrie.git#egg=pysettrie
pip install ppopt[optional]
```
## Completed Features

- Solver interface for mpLPs and mpQPs with the following algorithms
  1. Serial and Parallel Combinatorial Algorithm
  2. Serial and Parallel Geometrical Algorithm
  3. Serial and Parallel Graph based Algorithm
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

For more information about Multiparametric programming and it's applications, [this paper](https://www.frontiersin.org/articles/10.3389/fceng.2020.620168/full) is a good jumping point.

## Quick Overview

To give a fast primer of what we are doing, we are solving multiparametric programming problems (fast) by writting parallel algorithms efficently. Here is a quick scaleing analysis on a large multiparametric program with the combinatorial algorithm.

![image](https://github.com/TAMUparametric/PPOPT/blob/main/Figures/loglog_scaling.png)
![image](https://github.com/TAMUparametric/PPOPT/blob/main/Figures/scaleing_eff.png)

Here is a benchmark against the state of the art multiparametric programming solvers. All tests run on the Terra Supercomputer at Texas A&M University. Matlab 2021b was used for solvers written in matlab and Python 3.8 was used for PPOPT.

![image](https://github.com/TAMUparametric/PPOPT/blob/main/Figures/bench.png)


