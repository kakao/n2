# Installation
`master` brunch is always lastest release version of N2. Let's start clone it.
`$> git clone https://some.where.over.the.rainbow/n2.git`
Then, update sub modules `git submodule update --init`.

## Python
`sudo python setup.py install` that's it. You may want to run unit-test of Python by issue the following command: `make test_python`

## C++
1. make static_lib
	- if you need shared library, then type `make shared_lib`
2. make install
	- you can specify where to install n2 with PREFIX environment value. default path is `/usr/local/`.
3. make test_cpp  # unit test

## Go
1. Set GOPATH first!
2. make go
