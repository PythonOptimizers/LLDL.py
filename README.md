# LLDL.py
Limited-memory LDL factorization in Python

[![Build Status](https://travis-ci.com/PythonOptimizers/qr_mumps.py.svg?token=33z5zptBt5SzXC4ZvLpF&branch=without-cython)](https://travis-ci.com/PythonOptimizers/qr_mumps.py)

Cython/Python inferface to qr_mumps ([A multithreaded multifrontal QR solver](http://buttari.perso.enseeiht.fr/qr_mumps/)).

It supports all four types (single real, double real, single complex and double complex).

## Dependencies

For the Python version:

- [`Numpy`](http://www.numpy.org)

For the Cython version, include everything needed for the Python version and add:

- [`Cython`](http://cython.org/)
- [`cygenja`](https://github.com/PythonOptimizers/cygenja).

To run the tests:

- nose.

## Optional dependencies

`LLDL.py` provides facilities for sparse matrices coming from the [`CySparse`](https://github.com/PythonOptimizers/cysparse) library.
If you want to use these facilities, set the location of the `CySparse` library in your `site.cfg` file.


## Installation

1. Clone this repo
    ```bash
    git clone https://github.com/PythonOptimizers/NLP.py
    ```

2. Copy `site.template.cfg` to `site.cfg` and modify `site.cfg` to match your configuration
    ```bash
    cp site.template.cfg site.cfg
    ```

3. Install `LLDL.py`

   - Python version
            ```bash
            python setup.py build
            python setup.py install [--prefix=...]
            ```

   - Cython version
            ```bash
            python generate_code.py
    	    python setup.py build
    	    python setup.py install [--prefix=...]
            ```
