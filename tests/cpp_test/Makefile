CXX ?= g++
CXXFLAGS += -O3 -march=native -std=c++11 -pthread -fPIC -fopenmp
CXXFLAGS += -I../../include/ -I../../third_party/spdlog/include/ -I../../third_party/googletest/googletest/ -I../../third_party/googletest/googletest/include/
LDFLAGS += -lpthread -L../../build/lib/static -ln2 -fopenmp

n2_test: n2_test.o
	$(CXX) -o $@ ../../build/obj/gtest-all.o ../../build/obj/gtest_main.o  $? $(LDFLAGS)

n2_test.o: n2_test.cpp
	$(CXX) $(CXXFLAGS) -c $?

clean:
	rm -f *.o n2_test
