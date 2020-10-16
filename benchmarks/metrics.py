from scipy.spatial.distance import pdist as scipy_pdist


def knn_threshold(data, count, epsilon):
    return data[count - 1] + epsilon


def knn_recall(dataset_distances, run_distances, count, epsilon=1e-3):
    t = knn_threshold(dataset_distances, count, epsilon)
    actual = sum(d <= t for d in run_distances[:count])
    return actual / float(count)


def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]


metrics = {
    'euclidean': {
        'distance': lambda a, b: pdist(a, b, "euclidean")
    },
    'angular': {
        'distance': lambda a, b: pdist(a, b, "cosine")
    },
    'dot': {
        'distance': lambda a, b: - sum([a[i] * b[i] for i in range(len(a))])
    }
}
