from scipy.spatial.distance import pdist as scipy_pdist


def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]


metrics = {
    'euclidean': {
        'distance': lambda a, b: pdist(a, b, "euclidean")
    },
    'angular': {
        'distance': lambda a, b: pdist(a, b, "cosine")
    }
}
