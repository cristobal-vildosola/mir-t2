import pyflann


class Index:

    def __init__(self, dataset_q, dataset_r, algorithm="linear", checks=-1):
        self.flann = pyflann.FLANN()

        self.dataset_q = dataset_q
        self.dataset_r = dataset_r
        self.flann.build_index(dataset_r, algorithm=algorithm)
        self.results, self.dists = self.flann.nn_index(dataset_q, num_neighbors=1, cores=1, checks=checks)
