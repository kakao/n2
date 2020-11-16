Installation
==============================================================================

``master`` branch is always the latest release version of N2 and
``dev`` branch is the development branch for the next release.

Requirements
------------------------------------------------------------------------------

-  gcc
-  openmp

.. note::

   Note that you must install gcc that supports C++14.
   For macOS users, please ensure that gcc is installed with
   `brew <https://brew.sh/index.html>`__.
   Currently, N2 build is not supported for gcc linked to Clang.

.. note::

   Regardless of your language choice (Python, C++, Go), any command described
   in the section ``Install from source`` should be run from the root of N2 directory,
   assuming that you have successfully run the following commands.

   .. code:: bash

      $ git clone https://github.com/kakao/n2.git
      $ cd n2
      $ git submodule update --init  # update submodules

Python
------------------------------------------------------------------------------
You can install N2 using pip or directly from source.

Install using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The easiest way to install N2 is to use pip.
This will automatically install Cython dependency.

.. code:: bash

   $ pip install n2

Install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Or you can build from source by running the following command.

.. code:: bash

   $ python setup.py install

You can run unit test with:

.. code:: bash
   
   $ make test_python

C++
------------------------------------------------------------------------------
Install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Depending on what you want, run either of the following commands:

   .. code:: bash

      $ make shared_lib  # If you need shared library

   .. code:: bash

      $ make static_lib  # If you need static library

2. You can install N2 shared library (built with ``make shared_lib``)
into user-defined location set by PREFIX environment variable with the following command:

   .. code:: bash

      $ make install  # Default installation path is /usr/local/.

3. You can run unit test with:

   .. code:: bash

      $ make test_cpp

Go
------------------------------------------------------------------------------
Install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Set GOPATH first!
   $ make go

Installation FAQ
------------------------------------------------------------------------------
I'm having trouble installing N2 on macOS.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After you install gcc with brew, ``python setup.py install`` will work fine.
But ``make shared_lib`` or ``make static_lib`` can still produce errors similar to the following:

.. code:: bash

   $ make shared_lib
   cd src/ && make shared_lib && cd .. && mkdir -p build/lib && \
         mv src/libn2.so ./build/lib/libn2.so && \
         cp build/lib/libn2.so build/lib/libn2.so.0.1.6
   c++ -O3 -march=native -std=c++14 -pthread -fPIC -fopenmp -DNDEBUG -DBOOST_DISABLE_ASSERTS
   -I../third_party/spdlog/include/ -I../include/ -I../third_party/eigen -I../third_party/boost/assert/include/
   -I../third_party/boost/bind/include/ -I../third_party/boost/concept_check/include/
   -I../third_party/boost/config/include/ -I../third_party/boost/core/include/ -I../third_party/boost/detail/include/
   -I../third_party/boost/heap/include/ -I../third_party/boost/iterator/include/ -I../third_party/boost/mp11/include/
   -I../third_party/boost/mpl/include/ -I../third_party/boost/parameter/include/
   -I../third_party/boost/preprocessor/include/ -I../third_party/boost/static_assert/include/
   -I../third_party/boost/throw_exception/include/ -I../third_party/boost/type_traits/include/
   -I../third_party/boost/utility/include/   -c -o hnsw.o hnsw.cc
   clang: error: unsupported option '-fopenmp'
   make[1]: *** [hnsw.o] Error 1
   make: *** [shared_lib] Error 2

In this case, possible reason is that you have not properly set symbolic links
or environment variables to point to brew-installed gcc.
Thus, please make sure that gcc/g++ symbolic links are linked to
brew-installed gcc, or CC/CXX environment variables are set as brew-installed gcc/g++.
There may be other solutions and here is one possible fix to this problem.

.. code:: bash

   # Set CC, CXX environment variables
   $ export CC=$(find $(brew --prefix gcc)/bin -type f -name 'gcc-[0-9]*')
   $ export CXX=$(find $(brew --prefix gcc)/bin -type f -name 'g++-[0-9]*')
