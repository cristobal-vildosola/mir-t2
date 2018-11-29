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


def obtener_curva(indice, queryset, lscan_time, lscan_dists, verbose=False):
    dists, time = indice.search(queryset, checks=1)
    
    efectividad = [evaluar_resultado(dists, lscan_dists)]
    eficiencia = [time / lscan_time]

    if verbose:
        print("Punto1 = {:.1f}%, {:.1f}% ({:.1f}s)".format(efectividad[-1], eficiencia[-1], time))

    checks = 2
    i = 2
    while efectividad[-1] < 99.9:
        # realizar busqueda
        dists, time = indice.search(queryset, checks=checks)

        # guardar efectividad y eficiencia
        efectividad.append(evaluar_resultado(dists, lscan_dists))
        eficiencia.append(time / lscan_time)

        if verbose:
            print("Punto{:d} = {:.1f}%, {:.1f}T ({:.1f}s)".format(i, efectividad[-1], eficiencia[-1], time))

        # aumentar número de checks
        checks *= 2
        i += 1

    return efectividad, eficiencia


def graficar_curvas(curvas, leyenda, titulo):
    plt.figure(figsize=(20, 7))

    # curvas
    plt.subplot(1, 2, 1)

    # graficar curvas y obtener límite en y
    max_efic = 1.0
    for efectividad, eficiencia in curvas:
        p = plt.plot(efectividad, eficiencia, 'o-')
        max_efic = max(max_efic, max(eficiencia))

    # configuración
    plt.xlabel('Efectividad')
    plt.ylabel('Eficiencia')

    plt.xlim([0, 100])
    plt.ylim([0, max_efic])

    plt.legend(leyenda)
    plt.title(titulo)

    # tabla
    plt.subplot(1, 2, 2)

    # obtener número de filas de la tabla
    rows = 0
    for efectividad, _ in curvas:
        rows = max(rows, len(efectividad))

    # generar texto de la tabla
    cells_text = [[] for _ in range(rows)]
    for efectividad, eficiencia in curvas:

        # agregar datos de la columna
        for i in range(len(efectividad)):
            cells_text[i].append('%1.1f%% / %1.1fT' % (efectividad[i], eficiencia[i]))

        # rellenar con vacío
        for i in range(len(efectividad), rows):
            cells_text[i].append('')

    # esconder gráfico vacío
    plt.axis('tight')
    plt.axis('off')

    # graficar tabla
    colors = ["C%i" % i for i in range(len(leyenda))]
    table = plt.table(cellText=cells_text,
                      colLabels=leyenda,
                      colColours=colors,
                      rowLabels=["Punto %i" % i for i in range(rows)],
                      loc='center left')

    # agrandar tamaño letra
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.4, 1.4)

    plt.show()
    return


def graficar_histograma(x, titulo, bins=10):
    plt.figure()

    plt.hist(x, bins)

    plt.xlabel('Distancia')
    plt.ylabel('Ocurrencias')

    plt.title(titulo)

    plt.show()
    return
