import numpy
import os
from scipy.spatial import distance
from Results import graficar_histograma


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


def dimension_intrinseca(dataset, titulo, porc_muestras):
    n = dataset.shape[0]
    muestras = int(n * porc_muestras)

    x_set = numpy.random.permutation(dataset)[:muestras]
    y_set = numpy.random.permutation(dataset)[:muestras]

    distancias = numpy.zeros(muestras ** 2)

    for i in range(muestras):
        for j in range(muestras):
            distancias[i + j * muestras] = distance.euclidean(x_set[i], y_set[j])

    mean = numpy.mean(distancias)
    std = numpy.std(distancias)
    graficar_histograma(distancias, titulo, bins=20)

    return mean ** 2 / (2 * std ** 2)
