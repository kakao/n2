CXX ?= g++
CXXFLAGS += -O3 -march=native -std=c++14 -pthread -fPIC -fopenmp
CXXFLAGS += -I../third_party/spdlog/include/ -I../include/ -I../third_party/eigen \
			-I../third_party/boost/assert/include/ -I../third_party/boost/bind/include/ \
			-I../third_party/boost/concept_check/include/ -I../third_party/boost/config/include/ \
			-I../third_party/boost/core/include/ -I../third_party/boost/detail/include/ \
			-I../third_party/boost/heap/include/ -I../third_party/boost/iterator/include/ \
			-I../third_party/boost/mp11/include/ -I../third_party/boost/mpl/include/ \
			-I../third_party/boost/parameter/include/ -I../third_party/boost/preprocessor/include/ \
			-I../third_party/boost/static_assert/include/ -I../third_party/boost/throw_exception/include/ \
			-I../third_party/boost/type_traits/include/ -I../third_party/boost/utility/include/
LDFLAGS += -lpthread

all: shared_lib

shared_lib: libn2.so

libn2.so: hnsw.o hnsw_build.o hnsw_search.o hnsw_model.o hnsw_node.o heuristic.o mmap.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $(LDFLAGS) $?

static_lib: libn2.a

libn2.a: hnsw.o hnsw_build.o hnsw_search.o hnsw_model.o hnsw_node.o heuristic.o mmap.o
	ar rvs $@ $?

clean:
	rm -f *.o n2.so n2.a
