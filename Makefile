GO_BIN := ${shell which go}
CURRENT_DIR := ${PWD}
MAJOR := 0
MINOR := 1
MICRO := 4
VERSION := $(MAJOR).$(MINOR).$(MICRO)
PREFIX ?= /usr/local/
PREFIX_INCLUDE := $(PREFIX)/include
PREFIX := $(PREFIX)/lib64
CXX ?= g++

all: static_lib shared_lib test_cpp
### Bindings ### 

go:
	@mkdir -p ${GOPATH}/src/n2
	@cp bindings/go/* ${GOPATH}/src/n2
	@cp -r include/* ${GOPATH}/src/n2
	@cp src/* ${GOPATH}/src/n2
	@cp tests/golang_test/* ${GOPATH}/src/n2
	@cp -r third_party ${GOPATH}/src/n2/
	@cd ${GOPATH}/src/n2 && ${GO_BIN} get -t -v ...
	@cd ${GOPATH}/src/n2 && ${GO_BIN} build
	@cd ${GOPATH}/src/n2 && ${GO_BIN} test


### Libraries ###
shared_lib:
	cd src/ && make shared_lib && cd .. && mkdir -p build/lib && \
		mv src/libn2.so ./build/lib/libn2.so && \
		cp build/lib/libn2.so build/lib/libn2.so.$(VERSION)

static_lib:
	cd src/ && make static_lib && cd .. && mkdir -p build/lib/static && \
		mv src/libn2.a ./build/lib/static/

### Installation ###

install:
	if [ -a build/lib/libn2.so.$(VERSION) ] ; \
	then \
		install build/lib/libn2.so.$(VERSION) $(PREFIX) && \
		ln -s `which $(PREFIX)/libn2.so.$(VERSION)` $(PREFIX)/libn2.so.tmp && \
		mv $(PREFIX)/libn2.so.tmp $(PREFIX)/libn2.so ; \
		cp -r include/n2 $(PREFIX_INCLUDE) ; \
	fi;
	echo "Finished"


### Tests ###

test_all: test_cpp test_python

test_cpp: static_lib gtest
	cd tests/cpp_test/ && make && ./n2_test

test_python:
	nosetests --nologcapture --nocapture -vv tests/python_test/test_n2.py

gtest: gtest-all.o gtest_main.o

gtest-all.o:
	@mkdir -p build/obj
	$(CXX) -o build/obj/$@ -c -I./third_party/googletest/googletest/ -I./third_party/googletest/googletest/include/ ./third_party/googletest/googletest/src/gtest-all.cc

gtest_main.o:
	@mkdir -p build/obj
	$(CXX) -o build/obj/$@ -c -I./third_party/googletest/googletest/ -I./third_party/googletest/googletest/include/ ./third_party/googletest/googletest/src/gtest_main.cc

.PHONY: clean
clean:
	cd src && make clean
	cd tests/cpp_test && make clean
	rm -rf *.o
	rm -rf test_run
	rm -rf build
