# -------------------------------------------------------------------------
# Aim:
# Search natural density peaks
# -------------------------------------------------------------------------
# Input:
# A: the data set
# -------------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors


def NDP_Searching(A):
    # tic;
    N, dim = A.shape
    k = min(100, N)
    kdtree = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(A)  # KDTreeSearcher(A(:,:),'bucketsize',1); % 1 indicates using Euclidean distance
    distances, index = kdtree.kneighbors(A)  # [index,~] = knnsearch(kdtree,A(:,:),'k',100); % Return sorted indices and distances

    # Initialize basic data
    r = 1
    flag = 0
    nb = np.zeros(N, dtype=int)  # Number of natural neighbors
    # NNN = np.zeros((N, N))  # Natural neighbor set of each point
    count = 0   # Consecutive count where the number of points with zero natural nearest neighbors remains unchanged
    count1 = 0  # Previous count of data points with zero natural nearest neighbors
    count2 = 0  # Current count of data points with zero natural nearest neighbors

    # Search natural nearest neighbors
    while flag == 0:
        for i in range(N):
            k_neighbor = index[i, r]
            nb[k_neighbor] = nb[k_neighbor] + 1
            # RNN[k_neighbor, nb[k_neighbor]] = i
        r = r + 1
        # count2 = 0
        count2 = int(np.sum(nb == 0))  # [~,count2]=size(find(nb==0));
        # for i in range(N):
        #     if nb[i] == 0:
        #         count2 = count2 + 1

        # Count consecutive iterations where the number of points with nb(i) == 0 does not change
        if count1 == count2:
            count = count + 1
        else:
            count = 1

        if count2 == 0 or (r > 2 and count >= 2):  # Neighbor search termination condition
            flag = 1

        count1 = count2

    supk = r - 1           # lambda
    max_nb = int(np.max(nb))  # maxnb
    rho = np.zeros(N)
    Non = max_nb
    # Non = supk

    #############################################
    # compute the density
    for i in range(N):
        d = 0.0
        for j in range(Non + 1):
            x = index[i, j]
            dist = np.sqrt(np.sum((A[i, :] - A[x, :]) ** 2))
            d = d + dist
        rho[i] = Non / d if d != 0 else np.inf

    # [rho_sorted, ordrho] = sort(rho, 'descend'); % ordrho is the order of densities from largest to smallest
    local_core = np.zeros(N, dtype=int)  # Store the local core points of the n points

    ##################################################################
    for i in range(N):
        rep = i
        xrho = rho[rep]
        for j in range(1, supk + 1):
            if xrho < rho[index[i, j]]:
                xrho = rho[index[i, j]]
                rep = index[i, j]
        local_core[i] = rep

    # update the local_core according to UR
    visited = np.zeros(N, dtype=int)
    round_val = 0
    for k_val in range(N):
        if visited[k_val] == 0:
            parent = k_val
            round_val = round_val + 1
            while local_core[parent] != parent:
                visited[parent] = round_val
                parent = local_core[parent]
            local_core[visited == round_val] = parent

    #############################################################
    # obtain NDPs
    cluster_number = 0
    cl = np.zeros(N, dtype=int)
    cores = []
    for i in range(N):
        if local_core[i] == i:
            cluster_number = cluster_number + 1
            cores.append(i)
            cl[i] = cluster_number
    cores = np.array(cores, dtype=int)

    for i in range(N):
        cl[i] = cl[local_core[i]]

    # toc;
    ###############################################################
    # import matplotlib.pyplot as plt
    # plt.figure()
    # # Plot the core points and corresponding initial clustering result
    # plt.plot(A[:, 0], A[:, 1], '.')
    # for i in range(N):
    #     plt.plot([A[i, 0], A[local_core[i], 0]], [A[i, 1], A[local_core[i], 1]])
    # drawcluster2(A, cl, cluster_number + 1)
    # plt.plot(A[local_core, 0], A[local_core, 1], 'ro', markersize=8, markerfacecolor='r', markeredgecolor='r')
    # # plt.title('Result of searching supk nearest neighbors')

    return index, supk, nb, rho, local_core, cores, cl, cluster_number




