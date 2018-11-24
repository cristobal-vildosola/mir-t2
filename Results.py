def evaluar_resultado(dists, real_dists):
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

    print("efectividad={:.1f}%  correctas={} incorrectas={}".format(100 * correctas / (correctas + incorrectas),
                                                                    correctas, incorrectas))
