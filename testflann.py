import os
import time

import numpy
import pyflann


def load_file(filename, num_vectors, vector_dimensions):
    assert os.path.isfile(filename), "no existe archivo " + filename
    mat = numpy.fromfile(filename, dtype=numpy.float32)
    return numpy.reshape(mat, (num_vectors, vector_dimensions))


def load_dataset_pair(dirname, num_vectors_q, num_vectors_r, vector_dimensions):
    file_q = "{}/Q-{}_{}_4F.bin".format(dirname, num_vectors_q, vector_dimensions)
    file_r = "{}/R-{}_{}_4F.bin".format(dirname, num_vectors_r, vector_dimensions)
    data_q = load_file(file_q, num_vectors_q, vector_dimensions)
    data_r = load_file(file_r, num_vectors_r, vector_dimensions)
    return data_q, data_r


(dataset_q, dataset_r) = load_dataset_pair("descriptores/MEL128", 21573, 33545, 128)
print("Q={} R={}".format(dataset_q.shape, dataset_r.shape))

(dataset_q, dataset_r) = load_dataset_pair("descriptores/SIFT", 2886, 202088, 128)
print("Q={} R={}".format(dataset_q.shape, dataset_r.shape))

(dataset_q, dataset_r) = load_dataset_pair("descriptores/VGG19", 842, 10171, 4096)
print("Q={} R={}".format(dataset_q.shape, dataset_r.shape))

# crea un objeto flann
flann = pyflann.FLANN()

# construir el indice linear scan
t0 = time.time()
flann.build_index(dataset_r, algorithm="linear")
t1 = time.time()
print("construccion linear scan={:.1f}".format(t1 - t0))

# buscar el NN usando el ultimo indice construido (linear scan)
t0 = time.time()
lscan_results, lscan_dists = flann.nn_index(dataset_q, num_neighbors=1, cores=1)
t1 = time.time()
print("busqueda linear scan={:.1f}".format(t1 - t0))

# construir el indice kdtree
t0 = time.time()
flann.build_index(dataset_r, algorithm="kdtree", trees=1)
t1 = time.time()
print("construccion kdtree={:.1f}".format(t1 - t0))

# buscar el NN con el ultimo indice construido (kdtree)
t0 = time.time()
exact_results, exact_dists = flann.nn_index(dataset_q, num_neighbors=1, cores=1, checks=-1)
t1 = time.time()
print("busqueda exacta kdtree={:.1f}".format(t1 - t0))

# buscar aproximada del NN con el ultimo indice construido (kdtree)
t0 = time.time()
approx_results, approx_dists = flann.nn_index(dataset_q, num_neighbors=1, cores=1, checks=100)
t1 = time.time()
print("busqueda aproximada kdtree={:.1f}".format(t1 - t0))


def evaluar_resultado(results, dists):
    correctas = 0
    incorrectas = 0
    iguales = 0
    for i in range(len(lscan_results)):
        # comparar las distancias
        if dists[i] == lscan_dists[i]:
            correctas += 1
        elif dists[i] > lscan_dists[i]:
            incorrectas += 1
        else:
            assert False, "distancia erronea!"
        # comparar el NN
        if results[i] == lscan_results[i]:
            iguales += 1
    print("efectividad={:.1f}%  correctas={} incorrectas={} mismo-NN={}".format(
        100 * correctas / (correctas + incorrectas), correctas, incorrectas, iguales))


evaluar_resultado(exact_results, exact_dists)
evaluar_resultado(approx_results, approx_dists)
