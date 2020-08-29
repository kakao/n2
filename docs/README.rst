Sphinx Local Build
==================

Method 1
--------

.. code-block:: bash

    $> pip install -r docs/requirements.txt
    $> git submodule update --init
    $> CC=gcc-10 CXX=g++-10 pip install .
    $> cd docs && make clean && make html

Method 2
--------

.. code-block:: bash

    $> pip install -r docs/requirements.txt
    $> git submodule update --init
    $> CC=gcc-10 CXX=g++-10 python setup.py build_ext --inplace

    # Uncomment the lines 13 ~ 15 from docs/conf.py before running the following command.
    $> cd docs && make clean && make html
    

Then you can view the documentation page by opening the docs/_build/html/index.html.