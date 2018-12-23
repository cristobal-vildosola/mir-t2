from Data import load_dataset_pair
from Index import Linear, KDTree, KMeansTree
from Results import obtener_curva

(queryset, dataset) = load_dataset_pair("descriptores/MEL128", 21573, 33545, 128)
print("Q={} R={}".format(queryset.shape, dataset.shape))

# construir el indice linear scan y buscar los NN
linear = Linear(dataset)
print("construccion linear scan = {:.1f}".format(linear.build_time))
_, lscan_dists, lscan_time = linear.search(queryset)
print("busqueda linear scan = {:.1f}\n".format(lscan_time))

# KD-Tree
trees = 10

# construir el indice KD-Tree
kdtree = KDTree(dataset, trees=trees)
print('construccion {:d}-KDTree = {:.1f}'.format(trees, kdtree.build_time))

# obtener curva y agregar al arreglo
_, eficiencia = obtener_curva(kdtree, queryset, lscan_time, lscan_dists, verbose=True)
print('{:d} busquedas {:d}-KDTree = {:.1f}\n'.format(len(eficiencia) - 1, trees, sum(eficiencia) * lscan_time))

# K-Means Tree
branches = 10

# construir el indice K-Means Tree
kmeanstree = KMeansTree(dataset, branching=branches)
print('construccion {:d}-KMeansTree = {:.1f}'.format(branches, kmeanstree.build_time))

# obtener curva y agregar al arreglo
_, eficiencia = obtener_curva(kmeanstree, queryset, lscan_time, lscan_dists, verbose=True)
print('{:d} busquedas {:d}-KMeansTree = {:.1f}\n'.format(
    len(eficiencia) - 1, branches, sum(eficiencia) * lscan_time))

print('Test terminado, todo funciona.')
