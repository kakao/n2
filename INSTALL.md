# Installation

`master` brunch is always lastest release version of N2. Let's start clone it.
`$> git clone https://some.where.over.the.rainbow/n2.git`
Then, update sub modules `git submodule update --init`.

## Python
The easiest way is to use pip. Simply do `sudo pip install n2` to fetch the package from [Python Package Index](https://pypi.org/)(PyPI). This will also install cython dependency.

For MacOS users, please set $CXX and $CC to your gcc/g++ path. e.g:
```
export CXX=/usr/local/bin/g++-7
export CC=/usr/local/bin/g++-7
```

You can also build from source by `python setup.py install`. You may want to run unit-test by issue the following command: `make test_python`.

## C++
1. make static_lib
	- if you need shared library, then type `make shared_lib`
2. make install
	- you can specify where to install n2 with PREFIX environment value. default path is `/usr/local/`.
3. make test_cpp  # unit test

## Go

1. Set GOPATH first!
2. make go


## Requirements
- gcc
- openmp
