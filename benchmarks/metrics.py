from scipy.spatial.distance import pdist as scipy_pdist


def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]


def knn_threshold(data, count, epsilon):
    return data[count - 1] + epsilon


def knn_recall(dataset_distances, run_distances, count, epsilon=1e-3):
    t = knn_threshold(dataset_distances, count, epsilon)
    actual = 0
    for d in run_distances[:count]:
        if d <= t:
            actual += 1
    return actual / float(count)


metrics = {
    'euclidean': {
        'distance': lambda a, b: pdist(a, b, "euclidean")
    },
    'angular': {
        'distance': lambda a, b: pdist(a, b, "cosine")
    }
}
