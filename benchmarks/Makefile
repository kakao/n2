CXX = g++
CXXFLAGS = -std=c++11 -fopenmp  -I../src -O3 -mavx -Wall
#CXXFLAGS = -std=c++11 -fopenmp -I /usr/local/include/eigen3 -I./src -O3 -Wall
CXXFLAGS += -I /usr/local/include/eigen3 -I../third_party/json11/ -I../third_party/spdlog/include -I../third_party/catch/include

HNSW_OBJ = base.o distance.o heuristic.o hnsw.o

BINARYDIR = bin

all: bin

search: $(BINARYDIR)/test_search_speed

#bin: $(BINARYDIR)/test_model $(BINARYDIR)/test_mmap $(BINARYDIR)/test_search_speed $(BINARYDIR)/unittest_base $(BINARYDIR)/unittest_hnsw
bin: $(BINARYDIR)/test_model $(BINARYDIR)/test_mmap $(BINARYDIR)/test_search_speed

$(BINARYDIR)/test_model: test/test_model.cc $(HNSW_OBJ) evaluator.o test_utils.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

$(BINARYDIR)/test_mmap: test/test_mmap.cc $(HNSW_OBJ) evaluator.o test_utils.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

$(BINARYDIR)/test_search_speed: test/test_search_speed.cc $(HNSW_OBJ) evaluator.o test_utils.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

$(BINARYDIR)/unittest_base: test/unittest_base.cc $(HNSW_OBJ) test_utils.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

$(BINARYDIR)/unittest_hnsw: test/unittest_hnsw.cc $(HNSW_OBJ) test_utils.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

evaluator.o: test/evaluator.cc test/evaluator.h
	$(CXX) -c -o $@ $< -fPIC $(CXXFLAGS)

test_utils.o: test/utils.cc test/utils.h
	$(CXX) -c -o $@ $< -fPIC $(CXXFLAGS)

hnsw.o: ../src/hnsw.cc ../src/hnsw.h ../src/mmap.h ../src/dy_heap.h ../src/hnsw_node.hpp ../src/sort.hpp
	$(CXX) -c -o $@ ../src/hnsw.cc -fPIC $(CXXFLAGS)

distance.o: ../src/distance.cc ../src/distance.h
	$(CXX) -c -o $@ $< -fPIC $(CXXFLAGS)

base.o: ../src/base.cc ../src/base.h
	$(CXX) -c -o $@ $< -fPIC $(CXXFLAGS)

heuristic.o: ../src/heuristic.cc ../src/heuristic.h
	$(CXX) -c -o $@ $< -fPIC $(CXXFLAGS)

.PHONY: clean

clean:
	rm -f *.o
	rm -f $(BINARYDIR)/*
