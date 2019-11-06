# ANN Benchmark 

## How to run

### #1. Download dataset
`$python download_dataset.py --dataset {name}`

#### List of dataset
* fashion-mnist-784-euclidean
* gist-960-euclidean
* glove-25-angular
* glove-50-angular
* glove-100-angular
* glove-200-angular
* mnist-784-euclidean
* random-xs-20-euclidean
* random-s-100-euclidean
* random-xs-20-angular
* random-s-100-angular
* sift-128-euclidean
* nytimes-256-angular
* youtube-40-angular
* youtube1m-40-angular

### #2. Run benchamark
`$python benchmark_script.py `

Result file will be created in './result' directory.

#### Parameters
* `--distance (euclidean or angular)`: 'Distance metric'
* `--dataset (sift or glove or youtube)`: 'Which dataset'
* `--n_threads (number of threads)`: 'Number of threads'

### #3. Visualize
`$python visualize.py {result_path}`
