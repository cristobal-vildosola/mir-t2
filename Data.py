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


def punto_debajo_de_curva(x, y, curva):
    # antes del primer segmento, se conecta con 0,0
    if x < curva[0][0]:
        x1, y1 = curva[0][0], curva[1][0]

        m = y1 / x1
        return y < m * x

    for i in range(1, len(curva[0])):

        # punto entre i - 1 e i
        if x < curva[0][i]:
            x1, y1 = curva[0][i - 1], curva[1][i - 1]
            x2, y2 = curva[0][i], curva[1][i]

            m = (y2 - y1) / (x2 - x1)
            return y < m * (x - x1) + y1

    # punto después del último segmento, se continua linea
    x1, y1 = curva[0][-2], curva[1][-2]
    x2, y2 = curva[0][-1], curva[1][-1]

    m = (y2 - y1) / (x2 - x1)
    return y < m * (x - x1) + y1


def largo(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 1 / 2


def curva_debajo_de_curva(curva1, curva2):
    # retorna un score que corresponde al porcentaje del largo de la curva1 que se encuentra debajo de la curva2.
    # para calular el largo debajo de la curva se toma segmento a segmento, si los 2 puntos del segmento
    # están debajo de la curva se suma el largo total, si 1 de los 2 se encuentra debajo se suma la mitad
    # del largo, y si ninguno está por debajo no se suma nada.

    largo_debajo = 0
    largo_total = 0

    anterior_debajo = punto_debajo_de_curva(curva1[0][0], curva1[1][0], curva2)

    for i in range(1, len(curva1[0])):
        largo_segmento = largo(curva1[0][i - 1], curva1[1][i - 1], curva1[0][i], curva1[1][i])
        largo_total += largo_segmento

        if punto_debajo_de_curva(curva1[0][i], curva1[1][i], curva2):
            if anterior_debajo:
                largo_debajo += largo_segmento
            else:
                largo_debajo += largo_segmento / 2

            anterior_debajo = True

        else:
            if anterior_debajo:
                largo_debajo += largo_segmento / 2

            anterior_debajo = False

    return largo_debajo / largo_total


def mejor_curva(curvas):
    mejor_i = 0

    for i in range(1, len(curvas)):
        score = curva_debajo_de_curva(curvas[i], curvas[mejor_i])

        # si más de la mitad de la curva se encuentra bajo la otra, se toma como mejor curva.
        if score > 0.5:
            mejor_i = i

    return mejor_i
