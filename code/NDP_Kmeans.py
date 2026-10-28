# -------------------------------------------------------------------------
# Aim:
# K-means clustering with natural density peaks for discovering arbitrary-shaped clusters
# -------------------------------------------------------------------------
# Input:
# A: the data set
# num_clusters: the expected number of clusters
# alpha: the proportion of noises [0,0.2]
# -------------------------------------------------------------------------
# output:
# cl2: the cluster label of each point n*1

import time
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path as csgraph_shortest_path
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


from NDP_Searching import NDP_Searching
from KmeansWithGD import KmeansWithGD
from drawcluster_ndpkmeans import drawcluster_ndpkmeans


def NDP_Kmeans(A, num_clusters, alpha, save_name='cluster_result.jpg'):
    t_start = time.time() 
    N, dim = A.shape
    print('Searching NDPs......')
    index, supk, nb, rho, local_core, cores, cl, cluster_number = NDP_Searching(A)
    rho_sorted = np.sort(rho)
    if alpha == 0:
        rho_threshold = 0.0
    else:
        rho_threshold = rho_sorted[int(np.floor(N * alpha))]

    ##################################################################
    # eliminate low density NDPs
    for i in range(cluster_number):
        if cores[i] != -1:  # cores(i)~=0
            if rho[cores[i]] < rho_threshold:  # Exclude core points with lower density
                mind = np.inf
                p = -1
                for j in range(cluster_number):
                    if i != j:
                        x = A[cores[i], :]
                        y = A[cores[j], :]
                        distance = np.sqrt(np.sum((x - y) ** 2))
                        if mind > distance and rho[cores[j]] > rho_threshold:
                            mind = distance
                            p = j
                if p != -1:
                    for j in range(N):
                        if local_core[j] == cores[i]:
                            local_core[j] = cores[p]

    cluster_number = 0
    cl = np.zeros(N, dtype=int)
    cores2 = []
    for i in range(N):
        if local_core[i] == i:
            cluster_number = cluster_number + 1
            cores2.append(i)
            cl[i] = cluster_number
    cores2 = np.array(cores2, dtype=int)

    print('The number of NDPs is', cluster_number)
    for i in range(N):
        cl[i] = cl[local_core[i]]

    #####################################################################
    print('Computing GD matrix of NDPs......')
    cdata = [[] for _ in range(cluster_number)]     # Store the points contained in each cluster
    cdataexp = [[] for _ in range(cluster_number)]  # Store the points in each cluster and the k-nearest neighbors of each point
    nc = np.zeros(cluster_number, dtype=int)         # Store the number of points belonging to a core point
    ncexp = np.zeros(cluster_number, dtype=int)
    core_dist = np.zeros((cluster_number, cluster_number))
    for i in range(cluster_number):
        for j in range(i + 1, cluster_number):
            x = A[cores2[i], :]
            y = A[cores2[j], :]
            d = np.sqrt(np.sum((x - y) ** 2))
            core_dist[i, j] = d
            core_dist[j, i] = d

    ###########################################################################
    maxd = 10 * np.max(core_dist) if cluster_number > 1 else 0
    sd = np.zeros(cluster_number)
    for i in range(cluster_number):
        nc[i] = 0
        ncexp[i] = 0
        x = A[cores2[i], :]
        for j in range(N):
            if cl[j] == (i + 1) and rho[j] > rho_threshold:
                nc[i] = nc[i] + 1
                y = A[j, :]
                sd[i] = sd[i] + np.sqrt(np.sum((x - y) ** 2))
                ncexp[i] = ncexp[i] + 1
                cdata[i].append(j)
                cdataexp[i].append(j)

        orig_ncexp_i = ncexp[i]
        for j in range(orig_ncexp_i):
            x_pt = cdata[i][j]
            d2 = np.sqrt(np.sum((A[x_pt, :] - A[cores2[i], :]) ** 2))
            for k in range(1, supk + 1):
                kneighbor = index[x_pt, k]
                if iscontain(cdataexp[i], kneighbor) == 0 and rho[kneighbor] > rho_threshold:
                    ncexp[i] = ncexp[i] + 1
                    cdataexp[i].append(kneighbor)

    sharedCount = np.zeros((cluster_number, cluster_number))
    count = 0
    for i in range(cluster_number):
        for j in range(i + 1, cluster_number):
            inset1 = np.intersect1d(cdataexp[i], cdataexp[j])
            averho = np.sum(rho[inset1])
            numinset1 = len(inset1)
            sharedCount[i, j] = numinset1
            sharedCount[j, i] = numinset1
            if numinset1 == 0:
                core_dist[i, j] = maxd * (core_dist[i, j] + 1)
                core_dist[j, i] = core_dist[i, j]
            else:
                count = count + 1
                core_dist[i, j] = core_dist[i, j] / (averho * numinset1)
                core_dist[j, i] = core_dist[i, j]

    # Compute minimum spanning tree between local core points
    SD = csr_matrix(core_dist)
    ST = minimum_spanning_tree(SD)
    ST2 = ST.toarray()
    ST2 = ST2 + ST2.T
    short_path = csgraph_shortest_path(csr_matrix(ST2), directed=False)

    ############################################################
    print('cluster NDPs with the improved kmeans......')
    # K-means algorithm implementation
    core_rho = rho[cores2]
    core_cl = KmeansWithGD(A[cores2, :], 'random', num_clusters, short_path, core_rho)
    cl2 = np.zeros(N, dtype=int)
    cl2[cores2] = core_cl
    for i in range(N):
        cl2[i] = cl2[local_core[i]]

    t_elapsed = time.time() - t_start
    print(f'Elapsed time is {t_elapsed:.6f} seconds.')
    print('Program complete!')
    plt.figure()
    drawcluster_ndpkmeans(A, cl2, num_clusters, filename=save_name)
    return cl2


def iscontain(q, x):
    # Here q is a column vector / array
    flag = 0
    for val in q:
        if val == x:
            flag = 1
            break
    return flag



