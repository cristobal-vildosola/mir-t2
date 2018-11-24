from Data import load_dataset_pair
from Index import Linear, KDTree
from Results import obtener_curva, graficar_curva

(queryset, dataset) = load_dataset_pair("descriptores/MEL128", 21573, 33545, 128)
print("Q={} R={}".format(queryset.shape, dataset.shape))

(queryset, dataset) = load_dataset_pair("descriptores/SIFT", 2886, 202088, 128)
print("Q={} R={}".format(queryset.shape, dataset.shape))

(queryset, dataset) = load_dataset_pair("descriptores/VGG19", 842, 10171, 4096)
print("Q={} R={}".format(queryset.shape, dataset.shape))

# construir el indice linear scan
linear = Linear(dataset)
print("construccion linear scan={:.1f}".format(linear.build_time))

# buscar el NN usando linear scan
lscan_dists, lscan_time = linear.search(queryset)
print("busqueda linear scan={:.1f}".format(lscan_time))

# construir el indice kdtree
kdtree = KDTree(dataset, trees=2)
print("construccion kdtree={:.1f}".format(kdtree.build_time))

efectividad, eficiencia = obtener_curva(kdtree, queryset, lscan_time, lscan_dists)
print(efectividad, eficiencia)

graficar_curva(efectividad, eficiencia)
