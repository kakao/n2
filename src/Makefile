CXX ?= g++
CXXFLAGS += -O3 -march=native -std=c++11 -pthread -fPIC -fopenmp
CXXFLAGS += -I../third_party/spdlog/include/ -I../include/ -I../third_party/eigen -I../third_party/boost/
LDFLAGS += -lpthread

all: shared_lib

shared_lib: libn2.so

libn2.so: hnsw.o hnsw_node.o distance.o heuristic.o mmap.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $(LDFLAGS) $?

static_lib: libn2.a

libn2.a: hnsw.o hnsw_node.o distance.o heuristic.o mmap.o
	ar rvs $@ $?

clean:
	rm -f *.o n2.so n2.a
