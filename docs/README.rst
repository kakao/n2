Sphinx Local Build
==================

.. code-block:: bash
    # 1. Build n2 library locally. (this is needed as Sphinx extracts documentation from an importable module.)
    $ pip install -r docs/requirements.txt
    $ git submodule update --init

    # You can use either of the following commands.
    $ CC=gcc-10 CXX=g++-10 pip install .
    $ CC=gcc-10 CXX=g++-10 python setup.py build_ext --inplace
    
    # 2. Sphinx Build
    $ cd docs && make clean && make html

Then you can view the documentation page by opening the docs/_build/html/index.html.
