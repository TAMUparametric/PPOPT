from setuptools import setup, find_packages

__version__ = "1.0.0"

short_desc = (
    "Extensible Multiparametric Solver in Python"
)

with open('README.md') as f:
    long_description = f.read()

setup(
    name='ppopt',
    version=__version__,
    author='Dustin R. Kenefake',
    author_email='Dustin.Kenefake@tamu.edu',
    description=short_desc,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    extras_require=dict(tests=['pytest']),
    install_requires=[  "numpy",
                        "pypoman",
                        "matplotlib",
                        "scipy",
                        "numba",
                        "gurobipy",
                        "pytest",
                        "setuptools",
                        "psutil",
                        "pathos",
                        "plotly",
                        "cvxopt",
                        "quadprog"],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
