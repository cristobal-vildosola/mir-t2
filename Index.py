import time

import pyflann


class Linear:

    def __init__(self, dataset):
        self.flann = pyflann.FLANN()

        t0 = time.time()
        self.flann.build_index(dataset, algorithm="linear")
        t1 = time.time()

        self.build_time = t1 - t0

    def search(self, queryset):
        t0 = time.time()
        _, dists = self.flann.nn_index(queryset, num_neighbors=1, cores=1)
        t1 = time.time()

        return dists, t1 - t0


class KDTree:

    def __init__(self, dataset, trees):
        self.flann = pyflann.FLANN()

        t0 = time.time()
        self.flann.build_index(dataset, algorithm="kdtree", trees=trees)
        t1 = time.time()

        self.build_time = t1 - t0

    def search(self, queryset, checks):
        t0 = time.time()
        _, dists = self.flann.nn_index(queryset, checks=checks, num_neighbors=1, cores=1)
        t1 = time.time()

        return dists, t1 - t0


class KMeansTree:

    def __init__(self, dataset, branching):
        self.flann = pyflann.FLANN()

        t0 = time.time()
        self.flann.build_index(dataset, algorithm='kmeans', branching=branching, iterations=-1)
        t1 = time.time()

        self.build_time = t1 - t0

    def search(self, queryset, checks):
        t0 = time.time()
        _, dists = self.flann.nn_index(queryset, checks=checks, num_neighbors=1, cores=1)
        t1 = time.time()

        return dists, t1 - t0
