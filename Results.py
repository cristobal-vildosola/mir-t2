import matplotlib.pyplot as plt


def evaluar_resultado(dists, real_dists, verbose=False):
    correctas = 0
    incorrectas = 0
    for i in range(len(real_dists)):
        # comparar las distancias
        if dists[i] == real_dists[i]:
            correctas += 1
        elif dists[i] > real_dists[i]:
            incorrectas += 1
        else:
            assert False, "distancia erronea!"

    if verbose:
        print("efectividad={:.1f}%  correctas={} incorrectas={}".format(100 * correctas / (correctas + incorrectas),
                                                                        correctas, incorrectas))

    return 100 * correctas / (correctas + incorrectas)


def obtener_curva(indice, queryset, lscan_time, lscan_dists):
    efectividad = [0]
    eficiencia = [0]

    checks = 1
    while efectividad[-1] < 99.9:
        # realizar busqueda
        dists, time = indice.search(queryset, checks=checks)

        # guardar efectividad y eficiencia
        efectividad.append(evaluar_resultado(dists, lscan_dists))
        eficiencia.append(time / lscan_time)

        # aumentar número de check
        checks *= 2

    # asegurarse de obtener 100% de efectividad
    if efectividad[-1] != 100:
        dists, time = indice.search(queryset, checks=-1)
        efectividad.append(evaluar_resultado(dists, lscan_dists))
        eficiencia.append(time / lscan_time)

    return efectividad, eficiencia


def graficar_curvas(curvas, leyenda):
    plt.figure()
    plt.xlabel('Efectividad')
    plt.ylabel('Eficiencia')
    plt.grid(True)
    max_efic = 1.0

    for efectividad, eficiencia in curvas:
        plt.plot(efectividad, eficiencia, 'o')
        max_efic = max(max_efic, max(eficiencia))

    plt.xlim([0, 100])
    plt.ylim([0, 1])

    plt.legend(leyenda)
    plt.show()
    return
