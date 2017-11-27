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
