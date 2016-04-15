#!/usr/bin/env python
{% raw %}
"""LLDL.py: A Limited-memory LDL factorization in Python

LLDL.py is a Python implementation of a modification of [Lin and More's limited-memory Cholesky factorization](http://dx.doi.org/10.1137/S1064827597327334) code-named ICFS for symmetric positive definite matrices. LLDL implements a similar limited-memory scheme for symmetric indefinite matrices and thus also for symmetric quasi-definite matrices.


S. Arreckx   <sylvain.arreckx@gmail.com>
"""
{% endraw %}
# The file setup.py is automatically generated
# Generate it with
# python generate_code -s

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension

import numpy as np

import ConfigParser
import os
import copy

from codecs import open
from os import path

DOCLINES = __doc__.split("\n")


# HELPERS
def prepare_Cython_extensions_as_C_extensions(extensions):
    """
    Modify the list of sources to transform `Cython` extensions into `C` extensions.
    Args:
        extensions: A list of (`Cython`) `distutils` extensions.
    Warning:
        The extensions are changed in place. This function is not compatible with `C++` code.
    Note:
        Only `Cython` source files are modified into their `C` equivalent source files. Other file types are unchanged.
    """
    for extension in extensions:
        c_sources = list()
        for source_path in extension.sources:
            path, source = os.path.split(source_path)
            filename, ext = os.path.splitext(source)

            if ext == '.pyx':
                c_sources.append(os.path.join(path, filename + '.c'))
            elif ext in ['.pxd', '.pxi']:
                pass
            else:
                # copy source as is
                c_sources.append(source_path)

        # modify extension in place
        extension.sources = c_sources

lldl_config = ConfigParser.SafeConfigParser()
lldl_config.read('site.cfg')

numpy_include = np.get_include()

# Use Cython?
use_cython = lldl_config.getboolean('CODE_GENERATION', 'use_cython')
if use_cython:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
    except ImportError:
        raise ImportError("Check '%s': Cython is not properly installed." % mumps_config_file)

# DEFAULT
default_include_dir = lldl_config.get('DEFAULT',
                                      'include_dirs').split(os.pathsep)
default_library_dir = lldl_config.get('DEFAULT',
                                      'library_dirs').split(os.pathsep)

# Debug mode
use_debug_symbols = lldl_config.getboolean('CODE_GENERATION',
                                           'use_debug_symbols')

# OPTIONAL
build_cysparse_ext = False
if lldl_config.has_section('CYSPARSE'):
    build_cysparse_ext = True
    cysparse_rootdir = lldl_config.get('CYSPARSE',
                                       'cysparse_rootdir').split(os.pathsep)
    if cysparse_rootdir == '':
        raise ValueError("You must specify where CySparse source code is" +
                         "located. Use `cysparse_rootdir` to specify its" +
                         "path.")


##########################################################################
# EXTENSIONS
##########################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
if not use_debug_symbols:
    ext_params['extra_compile_args'] = ["-O2", '-std=c99',
                                        '-Wno-unused-function']
    ext_params['extra_link_args'] = []
else:
    ext_params['extra_compile_args'] = ["-g", '-std=c99',
                                        '-Wno-unused-function']
    ext_params['extra_link_args'] = ["-g"]

context_ext_params = copy.deepcopy(ext_params)
lldl_ext = []
{% for index_type in index_list %}
    {% for element_type in type_list %}
base_ext_params_@index_type@_@element_type@ = copy.deepcopy(ext_params)
lldl_ext.append(Extension(name="lldl.src.lldl_@index_type@_@element_type@",
                          sources=['lldl/src/lldl_@index_type@_@element_type@.pxd',
                                   'lldl/src/lldl_@index_type@_@element_type@.pyx'],
                          **base_ext_params_@index_type@_@element_type@))

# numpy_ext_params_@index_type@_@element_type@ = copy.deepcopy(ext_params)
# numpy_ext_params_@index_type@_@element_type@['include_dirs'].extend(lldl_include_dirs)
# lldl_ext.append(Extension(name="lldl.src.numpy_lldl_@index_type@_@element_type@",
#                           sources=['lldl/src/numpy_lldl_@index_type@_@element_type@.pxd',
#                                    'lldl/src/numpy_lldl_@index_type@_@element_type@.pyx'],
#                           **numpy_ext_params_@index_type@_@element_type@))

    {% endfor %}
{% endfor %}

if build_cysparse_ext:
{% for index_type in index_list %}
 {% for element_type in type_list %}
    cysparse_ext_params_@index_type@_@element_type@ = copy.deepcopy(ext_params)
    cysparse_ext_params_@index_type@_@element_type@['include_dirs'].extend(cysparse_rootdir)
    lldl_ext.append(Extension(name="lldl.src.cysparse_lldl_@index_type@_@element_type@",
                                   sources=['lldl/src/cysparse_lldl_@index_type@_@element_type@.pxd',
                                            'lldl/src/cysparse_lldl_@index_type@_@element_type@.pyx'],
                                   **cysparse_ext_params_@index_type@_@element_type@))

     {% endfor %}
 {% endfor %}


packages_list = ['lldl', 'lldl.src', 'tests']


# PACKAGE PREPARATION FOR EXCLUSIVE C EXTENSIONS
##########################################################################
# We only use the C files **without** Cython. In fact, Cython doesn't need to be installed.
if not use_cython:
    prepare_Cython_extensions_as_C_extensions(lldl_ext)

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

setup_args = {
    'name': 'LLDL.py',
    'version': '0.0.1',
    'description' : DOCLINES[0],
    'long_description' : "\n".join(DOCLINES[2:]),
    # Author details
{% raw %}
    'author': 'Sylvain Arreckx, Dominique Orban and Nikolaj van Omme',
{% endraw %}
    'maintainer': "Sylvain Arreckx",
{% raw %}
    'maintainer_email': "sylvain.arreckx@gmail.com",
{% endraw %}
    'summary': "Limited-memory LDL factorization in Python.",
    'url': "https://github.com/PythonOptimizers/LLDL.py.git",
    'download_url': "https://github.com/PythonOptimizers/LLDL.py.git",
    'license': 'LGPL',
    'classifiers': filter(None, CLASSIFIERS.split('\n')),
    'install_requires': ['numpy'],
    'ext_modules': lldl_ext,
    'package_dir': {"lldl": "lldl"},
    'packages': packages_list,
    'zip_safe': False}

if use_cython:
    setup_args['cmdclass'] = {'build_ext': build_ext}

setup(**setup_args)
