# ANN Benchmark 

## How to run
`$python benchmark_script.py`

## Youtube dataset
Since downloading YouTube data sets automatically is not yet supported
(if you can make it possible, your contribution is always welcome),
please follow the instructions below.

1. Please download youtube dataset from [the google drive](https://drive.google.com/open?id=1B3PWRTb8xol9fEkawVbpfitOsuwXkqss)
2. Unzip it and put the youtube dataset (youtube.txt) into the `./dataset/` directory.
3. Run python script. `python benchmark_script.py --dataset youtube`

## Parameters
* `--distance (euclidean or angular)`: 'Distance metric'
* `--dataset (sift or glove or youtube)`: 'Which dataset'
* `--ef_con (ef_con value)`: 'Ef_con value'
* `--ef_search (ef_search metric)`: 'ef_search'
* `--M (M value)`: 'M value'
* `--n_threads (number of threads)`: 'Number of threads'

# [Benchmark](/docs/benchmark.rst) Reproduce

## Warning
This benchmark is a script that reproduces all of the benchmark metrics we measure.
It takes a considerable amount of time to run all the libraries.

## How to run
```
$pip install n2 nmslib annoy
$python youtube_reproduce.py
```

## Result
`./results/youtube.txt`

## Parameters
* `--algo (n2 | annoy | nmslib)`: 'A name of library which t you want to run. If this parameter is not set, benchmark will run all libraries.'
* `--n_threads (number of threads)`: 'Number of threads'
