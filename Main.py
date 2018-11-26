from Data import load_dataset_pair
from Index import Linear, KDTree, KMeansTree
from Results import obtener_curva, graficar_curvas

# cargar datos
(dataset_q, dataset_r) = load_dataset_pair("descriptores/MEL128", 21573, 33545, 128)
print("Q={} R={}".format(dataset_q.shape, dataset_r.shape))

(queryset2, dataset2) = load_dataset_pair("descriptores/SIFT", 2886, 202088, 128)
print("Q={} R={}".format(queryset2.shape, dataset2.shape))

(queryset3, dataset3) = load_dataset_pair("descriptores/VGG19", 842, 10171, 4096)
print("Q={} R={}".format(queryset3.shape, dataset3.shape))

# construir el indice linear scan y buscar el NN
linear = Linear(dataset3)
print("construccion linear scan={:.1f}".format(linear.build_time))

lscan_dists, lscan_time = linear.search(queryset3)
print("busqueda linear scan={:.1f}".format(lscan_time))

curvas = []
leyenda = []

# obtener curvas para KD-Tree
num_trees = [5, 10, 20, 40, 60]
for trees in num_trees:
    # construir el indice KD-Tree
    kdtree = KDTree(dataset3, trees=trees)
    print('construccion {:d}-KDTree = {:.1f}'.format(trees, kdtree.build_time))

    # obtener curva y agregar al arreglo
    efectividad, eficiencia = obtener_curva(kdtree, queryset3, lscan_time, lscan_dists)
    curvas.append([efectividad, eficiencia])
    leyenda.append('KDTree con {:d} árboles'.format(trees))
    print('{:d} busquedas {:d}-KDTree = {:.1f}'.format(len(eficiencia) - 1, trees, sum(eficiencia) * lscan_time))

# graficar las curvas
graficar_curvas(curvas, leyenda)

curvas = []
leyenda = []

# obtener curvas para K-Means Tree
num_branches = [5, 10, 15, 20, 30]
for branches in num_branches:
    # construir el indice K-Means Tree
    kmeanstree = KMeansTree(dataset3, branching=branches)
    print('construccion {:d}-KMeansTree = {:.1f}'.format(branches, kmeanstree.build_time))

    # obtener curva y agregar al arreglo
    efectividad, eficiencia = obtener_curva(kmeanstree, queryset3, lscan_time, lscan_dists)
    curvas.append([efectividad, eficiencia])
    leyenda.append('KMeansTree con {:d} ramas'.format(branches))
    print('{:d} busquedas {:d}-KMeansTree = {:.1f}'.format(len(eficiencia) - 1, branches, sum(eficiencia) * lscan_time))

# graficar las curvas
graficar_curvas(curvas, leyenda)
