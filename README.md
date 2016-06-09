# LLDL.py
Limited-memory LDL factorization in Python

[![Build Status](https://travis-ci.com/PythonOptimizers/LLDL.py.svg?token=33z5zptBt5SzXC4ZvLpF&branch=master)](https://travis-ci.com/PythonOptimizers/LLDL.py)


Details of the implemented method are published in the folling paper:
[D. Orban. Limited-Memory LDL<sup>T</sup> Factorization of Symmetric Quasi-Definite Matrices with Application to Constrained Optimization. Cahier du GERAD G-2013-87. GERAD, Montreal, Canada](http://www.gerad.ca/~orban/_static/go2013.pdf).

## Dependencies

For the Python version:

- [`Numpy`](http://www.numpy.org)

For the Cython version, include everything needed for the Python version and add:

- [`Cython`](http://cython.org/)
- [`cygenja`](https://github.com/PythonOptimizers/cygenja).

To run the tests:

- pytest.

## Optional dependencies

`LLDL.py` provides facilities for sparse matrices coming from the [`CySparse`](https://github.com/PythonOptimizers/cysparse) library.


## Installation

### branch `master`

1. Clone this repo
    ```bash
    git clone https://github.com/PythonOptimizers/LLDL.py
    ```

2. Copy `site.template.cfg` to `site.cfg` and modify `site.cfg` to match your configuration
    ```bash
    cp site.template.cfg site.cfg
    ```

3. Install `LLDL.py`
    ```bash
    python setup.py build
    python setup.py install [--prefix=...]
    ```

### branch `develop`

Additionnal dependencies:
    ```bash
    pip install Cython
    pip install cygenja
    ```

1. Clone this repo
    ```bash
    git clone https://github.com/PythonOptimizers/LLDL.py
    ```

2. Copy `site.template.cfg` to `site.cfg` and modify `site.cfg` to match your configuration
    ```bash
    cp site.template.cfg site.cfg
    ```

3. Install `LLDL.py`
    ```bash
    python generate_code.py
    python setup.py build
    python setup.py install [--prefix=...]
    ```

### Running tests
    ```bash
    py.test -v tests
    ```
