Installation
==============================================================================

``master`` branch is always the latest release version of N2.

Requirements
------------------------------------------------------------------------------

-  gcc
-  openmp

.. note::

   Note that you must install gcc that supports C++14.
   For macOS users, please ensure that gcc is installed with
   `brew <https://brew.sh/index.html>`__.
   Currently, N2 build is not supported for gcc linked to Clang.

Python
------------------------------------------------------------------------------
You can install using pip or install directly from source.

1. Install using pip

The easiest way to install N2 is to use `pip`. This will automatically install Cython
dependency.

.. code:: bash

   $ pip install n2

2. Install from source

Or you can build from source by running the following commands.

.. code:: bash

   $ git clone https://github.com/kakao/n2.git
   $ git submodule update --init  # update submodules
   $ python setup.py install
   
3. You can run unit test with: ``make test_python``.

C++
------------------------------------------------------------------------------
You can install from source.

1. ``make shared_lib`` (if you need shared library) or
``make static_lib`` (if you need static library)

2. ``make install``
   - This installs N2 library built by ``make shared_lib`` into user-defined
   location set by PREFIX environment variable.
   - Default installation path is ``/usr/local/``.

3. You can run unit test with: ``make test_cpp``

Go
------------------------------------------------------------------------------
You can install from source.

1. Set GOPATH first!
2. ``make go``

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

In this case, possible reason is that you have not properly set symbolic links or environment variables to point to brew-installed gcc. Thus, please make sure that gcc/g++ symbolic links are linked to brew-installed gcc, or CC/CXX environment variables are set as brew-installed gcc/g++. There may be other solutions and here is one possible fix to this problem.

.. code:: bash

   # Set CC, CXX environment variables
   $ export CC=$(find $(brew --prefix gcc)/bin -type f -name 'gcc-[0-9]*')
   $ export CXX=$(find $(brew --prefix gcc)/bin -type f -name 'g++-[0-9]*')
