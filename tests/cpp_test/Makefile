CXX ?= g++
CXXFLAGS += -O3 -march=native -std=c++14 -pthread -fPIC -fopenmp
CXXFLAGS += -I../../include/ -I../../third_party/googletest/googletest/ -I../../third_party/googletest/googletest/include/
CXXFLAGS += -I../../third_party/spdlog/include/ -I../../include/ -I../../third_party/eigen \
			-I../../third_party/boost/assert/include/ -I../../third_party/boost/bind/include/ \
			-I../../third_party/boost/concept_check/include/ -I../../third_party/boost/config/include/ \
			-I../../third_party/boost/core/include/ -I../../third_party/boost/detail/include/ \
			-I../../third_party/boost/heap/include/ -I../../third_party/boost/iterator/include/ \
			-I../../third_party/boost/mp11/include/ -I../../third_party/boost/mpl/include/ \
			-I../../third_party/boost/parameter/include/ -I../../third_party/boost/preprocessor/include/ \
			-I../../third_party/boost/static_assert/include/ -I../../third_party/boost/throw_exception/include/ \
			-I../../third_party/boost/type_traits/include/ -I../../third_party/boost/utility/include/
LDFLAGS += -lpthread -L../../build/lib/static -ln2 -fopenmp

n2_test: n2_test.o
	$(CXX) -o $@ ../../build/obj/gtest-all.o ../../build/obj/gtest_main.o  $? $(LDFLAGS)

n2_test.o: n2_test.cpp
	$(CXX) $(CXXFLAGS) -c $?

clean:
	rm -f *.o n2_test
