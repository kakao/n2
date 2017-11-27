// Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package n2_test

import (
	"n2"
	"math/rand"
	"os"
	"testing"
)

func FloatArrayEquals(a []float32, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func IntArrayEquals(a []int, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func TestSearchByVectorL2(t *testing.T) {
	f := 3
	index := n2.NewHnswIndex(f, "L2")
	index.AddData([]float32{0, 0, 1})
	index.AddData([]float32{0, 1, 0})
	index.AddData([]float32{1, 0, 0})
	index.Build(5, 10, 4, 10, 3.5, "heuristic", "skip")
	index.SaveModel("test.n2")

	var result []int
	var distance []float32
	index.SearchByVector([]float32{1, 2, 3}, 3, -1, &result, &distance)
	if len(result) != 3 {
		t.Errorf("Result should be 3 not %d", len(result))
	}
	expected := []int{0, 1, 2}
	expected_distances := []float32{9.0, 11.0, 13.0}
	if IntArrayEquals(expected, result) != true {
		t.Errorf("Expected %v but got %v", expected, result)
	}

	if FloatArrayEquals(expected_distances, distance) != true {
		t.Errorf("Expected %v but got %v", expected_distances, distance)
	}

	n2.DeleteHnswIndex(index)
	index = n2.NewHnswIndex(f, "L2")
	if ret := index.LoadModel("test.n2"); ret == false {
		t.Errorf("Failed to load file")
	}

	n2.DeleteHnswIndex(index)

	index = n2.NewHnswIndex(f, "L2")
	if ret := index.LoadModel("test.n2", false); ret == false {
		t.Errorf("Failed to load file")
	}

	n2.DeleteHnswIndex(index)
	os.Remove("test.n2")
}

func TestSearchByVectorAngular(t *testing.T) {
	f := 3
	index := n2.NewHnswIndex(f, "angular")
	index.AddData([]float32{0, 0, 1})
	index.AddData([]float32{0, 1, 0})
	index.AddData([]float32{1, 0, 0})
	index.Build(5, 10, 4, 10, 3.5, "heuristic", "skip")

	var result []int

	expected1 := []int{0, 1, 2}
	expected2 := []int{0, 1, 2}
	expected3 := []int{0, 1, 2}
	index.SearchByVector([]float32{3, 2, 1}, 3, -1, &result)
	if len(result) != 3 {
		t.Errorf("The length of the result should be 3 not %d: %v", len(result), result)
	}

	if IntArrayEquals(expected1, result) != true {
		t.Errorf("Expected %v but got %v", expected1, result)
	}

	index.SearchByVector([]float32{1, 2, 3}, 3, -1, &result)
	if len(result) != 3 {
		t.Errorf("The length of the result should be 3 not %d: %v", len(result), result)
	}

	if IntArrayEquals(expected2, result) != true {
		t.Errorf("Expected %v but got %v", expected2, result)
	}

	index.SearchByVector([]float32{2, 0, 1}, 3, -1, &result)
	if len(result) != 3 {
		t.Errorf("Result should be 3 not %d", len(result))
	}

	if IntArrayEquals(expected3, result) != true {
		t.Errorf("Expected %v but got %v", expected3, result)
	}

	n2.DeleteHnswIndex(index)
}

func TestSearchByIdAngular(t *testing.T) {
	f := 3
	index := n2.NewHnswIndex(f, "angular")
	index.AddData([]float32{2, 1, 0})
	index.AddData([]float32{1, 2, 0})
	index.AddData([]float32{0, 0, 1})
	index.Build(5, 10, 4, 1, 3.5, "heuristic", "skip")

	var result []int

	expected1 := []int{0, 1, 2}
	expected2 := []int{1, 0, 2}
	index.SearchById(0, 3, -1, &result)
	if len(result) != 3 {
		t.Errorf("Result should be 3 not %d", len(result))
	}

	if IntArrayEquals(expected1, result) != true {
		t.Errorf("Expected %v but got %v", expected1, result)
	}

	index.SearchById(1, 3, -1, &result)
	if len(result) != 3 {
		t.Errorf("A length of the result should be 3 not %d", len(result))
	}

	if IntArrayEquals(expected2, result) != true {
		t.Errorf("Expected %v but got %v", expected2, result)
	}

	n2.DeleteHnswIndex(index)
}
func TestLargeL2(t *testing.T) {
	f := 10
	index := n2.NewHnswIndex(f, "L2")
	for j := 0; j < 10000; j += 2 {
		p := make([]float32, 0, 10)
		for i := 0; i < 10; i++ {
			p = append(p, rand.Float32())
		}
		x := make([]float32, 0, 10)
		for i := 0; i < 10; i++ {
			x = append(x, 1+p[i]+rand.Float32()*1e-2)
		}
		y := make([]float32, 0, 10)
		for i := 0; i < 10; i++ {
			y = append(y, 1+p[i]+rand.Float32()*1e-2)
		}
		index.AddData(x)
		index.AddData(y)
	}

	index.Build(5, 10, 300, 10, 3.5, "heuristic", "skip")

	for j := 0; j < 10000; j += 2 {
		var result []int
		index.SearchById(j, 2, -1, &result)
		if len(result) != 2 {
			t.Errorf("length of the result should be 2 not %d", len(result))
		}

		expected1 := []int{j, j + 1}
		if IntArrayEquals(expected1, result) != true {
			t.Errorf("Expected %v but got %v", expected1, result)
		}

		index.SearchById(j+1, 2, -1, &result)
		if len(result) != 2 {
			t.Errorf("length of the result should be 2 not %d", len(result))
		}
		expected2 := []int{j + 1, j}
		if IntArrayEquals(expected2, result) != true {
			t.Errorf("Expected %v but got %v", expected2, result)
		}
	}
	n2.DeleteHnswIndex(index)
}
