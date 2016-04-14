# LLDL.py
Limited-memory LDL factorization in Python

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
    git clone https://github.com/PythonOptimizers/LLDL.py
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
