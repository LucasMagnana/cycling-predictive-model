def cluster(X, clustering_method):
    clustering = clustering_method.fit(X)
    return clustering.labels_


def tab_clusters_to_dict(clusters):
    dict_cluster = {}
    for i in range(len(clusters)):
        if(clusters[i] in dict_cluster):
            dict_cluster[clusters[i]].append(i)
        else:
            dict_cluster[clusters[i]] = [i]
    return dict_cluster


