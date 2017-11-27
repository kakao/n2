package main

import (
    "n2"
    "math/rand"
    "fmt"
)

func main() {
     f := 3
     k := 3
     t := n2.NewHnswIndex(f, "L2")
     id := 2
     qvec := []float32{2, 1, 0}

     for i := 0; i < 1000; i++ {
       item := make([]float32, 0, f)
       for x:= 0; x < f; x++ {
           item = append(item, rand.Float32())
       }
       t.AddData(item)
     }
     t.Build(5, 10, 4, 10, 3.5, "heuristic", "skip")
     t.SaveModel("test.n2")

     other := n2.NewHnswIndex(0, "L2")
     other.LoadModel("test.n2")

     var result []int
     var distance []float32
     other.SearchByVector(qvec, k, -1, &result, &distance)
     fmt.Printf("[SearchByVector]: Neareast neighborhoods of %v: %v\n", qvec, result)
   
     other.SearchById(id, k, -1, &result, &distance)
     fmt.Printf("[SearchById]: Neareast neighborhoods of %v: %v\n", id, result)
}
