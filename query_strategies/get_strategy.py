from .kcenter_greedy import KCenterGreedy
from .random_sampling import RandomSampling
from .uncertainty_variance_dropout import UVarianceDropout
from .maxmin import MaxMin
from .kmeans_sampling import KMeansSampling


def get_strategy(args):
    name = args.strategy
    if name == 'kcenter':
        return KCenterGreedy
    elif name == 'random':
        return RandomSampling
    elif name == 'variance':
        return UVarianceDropout
    elif name == 'maxmin':
        return MaxMin
    elif name == 'kmean':
        return KMeansSampling
