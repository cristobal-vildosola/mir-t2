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
        results, dists = self.flann.nn_index(queryset, num_neighbors=1, cores=1)
        t1 = time.time()

        return results, dists, t1 - t0

    def all_nn(self, queryset):
        n = 113  # determinado utilizando test_equidistantes()
        nn = []
        max_repeat = 1

        # sigue buscando hasta asegurarse de encontrar todos los nn que estén a la misma distancia
        done = False
        while not done:

            max_repeat = 1
            done = True

            # buscar n vecinos cercanos
            results, dists = self.flann.nn_index(queryset, num_neighbors=n, cores=1)
            for i in range(len(results)):

                # agregar siempre el primer elemento
                nn.append([results[i][0]])

                # buscar todos los elementos que están a la misma distancia que el primero
                for j in range(1, n):
                    if dists[i][j] == dists[i][0]:
                        nn[i].append(results[i][j])
                        max_repeat = max((j + 1), max_repeat)

                    else:
                        break

                # si es que habían n elementos a la misma distancia, aumentar n y repetir busqueda
                if max_repeat == n:
                    done = False
                    n *= 2
                    nn = []
                    break

        return nn, max_repeat


class KDTree:

    def __init__(self, dataset, trees):
        self.flann = pyflann.FLANN()

        t0 = time.time()
        self.flann.build_index(dataset, algorithm="kdtree", trees=trees)
        t1 = time.time()

        self.build_time = t1 - t0

    def search(self, queryset, checks):
        t0 = time.time()
        results, dists = self.flann.nn_index(queryset, checks=checks, num_neighbors=1, cores=1)
        t1 = time.time()

        return results, dists, t1 - t0


class KMeansTree:

    def __init__(self, dataset, branching):
        self.flann = pyflann.FLANN()

        t0 = time.time()
        self.flann.build_index(dataset, algorithm='kmeans', branching=branching, iterations=-1)
        t1 = time.time()

        self.build_time = t1 - t0

    def search(self, queryset, checks):
        t0 = time.time()
        results, dists = self.flann.nn_index(queryset, checks=checks, num_neighbors=1, cores=1)
        t1 = time.time()

        return results, dists, t1 - t0


def test_equidistantes():
    from Data import load_dataset_pair

    (queryset1, dataset1) = load_dataset_pair("descriptores/MEL128", 21573, 33545, 128)
    (queryset2, dataset2) = load_dataset_pair("descriptores/SIFT", 2886, 202088, 128)
    (queryset3, dataset3) = load_dataset_pair("descriptores/VGG19", 842, 10171, 4096)

    linear = Linear(dataset1)
    _, max_repeat = linear.all_nn(queryset1)
    print(f'max equidistantes q1: {max_repeat}\n')

    linear = Linear(dataset2)
    _, max_repeat = linear.all_nn(queryset2)
    print(f'max equidistantes q2: {max_repeat}\n')

    linear = Linear(dataset3)
    _, max_repeat = linear.all_nn(queryset3)
    print(f'max equidistantes q3: {max_repeat}\n')

    return


if __name__ == '__main__':
    test_equidistantes()
