Installation
==========================================================================

``master`` brunch is always the latest release version of N2.

Python
---------------------------------------------------------------------

Install using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install N2 is to use pip. Simply run ``pip install n2`` to
fetch the package from `Python Package
Index <https://pypi.org/>`__\ (PyPI). This will also install Cython
dependency.

For MacOS users, please install gcc >= 7.0 with `brew <https://brew.sh/index.html>`__ to install gcc.

.. code:: bash

    $ brew install gcc
    $ sudo pip install n2

Install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also build from source by running the following commands.

.. code:: bash

   $ git clone https://github.com/kakao/n2.git
   $ pip install Cython
   $ git submodule update --init  # update submodules
   $ python setup.py install
   
You may want to run unit-test by issuing the following command:
``make test_python``.

C++
---------------------------------------------------------------------

1. ``make static_lib`` (for static library) or ``make shared_lib`` (for shared library)

2. ``make install``

   -  you can specify where to install n2 with PREFIX environment value.
      default path is ``/usr/local/``.

3. ``make test_cpp``  # unit test

Go
--

1. Set GOPATH first!
2. ``make go``

Requirements
---------------------------------------------------------------------

-  gcc
-  openmp
-  spdlog
