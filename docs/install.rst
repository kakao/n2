Installation
============

``master`` brunch is always lastest release version of N2. Let’s start
clone it. ``$> git clone https://github.com/kakao/n2.git`` Then, update
sub modules ``git submodule update --init``.

Python
------

Install using pip
*****************

The easiest way is to use pip. Simply do ``pip install n2`` to
fetch the package from `Python Package
Index <https://pypi.org/>`__\ (PyPI). This will also install cython
dependency.

For MacOS users, please install gcc >= 7.0 with `brew <https://brew.sh/index.html>`__ to install gcc.

::

    $> brew install gcc
    $> sudo pip install n2

Install from source
*******************

You can also build from source by running the following commands.

::

   $> pip install Cython
   $> git submodule update --init
   $> python setup.py install
   
You may want to run unit-test by issuing the following command:
``make test_python``.

C++
---

1. make static_lib

   -  if you need shared library, then type ``make shared_lib``

2. make install

   -  you can specify where to install n2 with PREFIX environment value.
      default path is ``/usr/local/``.

3. make test_cpp # unit test

Go
--

1. Set GOPATH first!
2. make go

Requirements
------------

-  gcc
-  openmp
-  spdlog
