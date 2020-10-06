Sphinx Local Build
==============================================================================

.. code-block:: bash

    # 1. Build n2 library locally.
    (this is needed as Sphinx extracts documentation from an importable module.)
    # 1-1. Build Preparation
    $ pip install -r docs/requirements.txt
    # install doxygen
    $ git submodule update --init

    # 1-2. Build n2. You can use either of the following commands.
    $ CC=gcc-10 CXX=g++-10 pip install .
    $ CC=gcc-10 CXX=g++-10 python setup.py build_ext --inplace
    
    # 2. Sphinx Build
    $ cd docs && make clean && make html

(side note) Instead of running ``1-2. and 2.``,
you can run ``make n2 && make clean && make html`` inside docs/ directory.

Then you can view the documentation page by opening the docs/_build/html/index.html.
