# -------------------------------------------------------------------------
# Aim:
# cluster NDPs with improved kmeans
# -------------------------------------------------------------------------

import numpy as np
def KmeansWithGD(data, centers, num_clusters, graphdist, rho):
    # dist is the shortest-path-based distance between core points
    # A is the local core points / density peaks, Ncores*d
    # cl is the kmeans clustering result, class label of each local core point
    #
    #   Input    : data           : Local core points / density peaks, Ncores*d,
    #                               D is the number of dimensions
    #              centers        : K-by-D matrix, where K is num_clusters, or
    #                               'random', random initialization, or
    #                               [], empty matrix, orthogonal initialization
    #              num_clusters   : Number of clusters
    #
    #   Output   : cluster_labels : N-by-1 vector of cluster assignment
    #
    # Parameter setting
    #
    # tic;
    iter = 0
    qold = np.inf
    threshold = 0.00000001

    #
    # Check if with initial centers
    #
    if isinstance(centers, str) and centers == 'random':
        centers = plus_init(graphdist, num_clusters, rho)
    elif centers is None or (hasattr(centers, '__len__') and len(centers) == 0):
        # disp('Orthogonal initialization...');
        centers = orth_init(data, num_clusters)
    else:
        centers = np.array(centers, dtype=int).copy()

    n = data.shape[0]
    # Algorithm iteration based on graph distance
    P = graphdist[centers, :]
    # fprintf('Initial cluster centers are:')
    while iter <= 100:
        iter = iter + 1
        # Find the closest cluster for each data point, i.e., partition non-center points
        # [val, ind] = min(P, [], 1);
        ind = np.argmin(P, axis=0)
        # input('Press enter');
        # drawcluster2(data,ind,num_clusters);
        # update centers
        clusters = [None] * num_clusters
        for i in range(num_clusters):
            clusters[i] = np.where(ind == i)[0]
            # Compute the sum of distances from each point in the i-th cluster to other points
            mindist = np.inf
            mind = centers[i]
            for j in range(len(clusters[i])):
                dis = np.sum(graphdist[clusters[i][j], clusters[i]])
                if mindist > dis:
                    mindist = dis
                    mind = clusters[i][j]
            centers[i] = mind

        #####################################################################
        # update members
        P = graphdist[centers, :]
        # Calculate objective function value
        qnew = float(np.sum(P[ind, np.arange(n)]))
        # mesg = sprintf('Iteration %d:\n\tQold=%g\t\tQnew=%g', iter, full(qold), full(qnew));
        # disp(mesg);

        # Check if objective function value is less than/equal to threshold
        if qold != np.inf and threshold >= abs((qnew - qold) / qold):
            # mesg = sprintf('\nkmeans converged!');
            # disp(mesg);
            break
        qold = qnew

    ######################################################################
    cluster_labels = ind + 1  # 1-based cluster assignment to match MATLAB
    ncluster = np.max(cluster_labels)
    return cluster_labels


# -----------------------------------------------------------------------------
def random_init(data, num_clusters):
    # RANDOM_INIT Initialize centroids choosing num_clusters rows of data at random
    #
    #   Input : data         : N-by-D data matrix, where N is the number of data,
    #                          D is the number of dimensions
    #           num_clusters : Number of clusters
    #
    #   Output: init_centers : K-by-D matrix, where K is num_clusters
    idx = np.random.randint(0, data.shape[0], size=num_clusters)
    init_centers = data[idx, :]
    return init_centers


# --------------------------------------------------------------------------
def random_init2(data, num_clusters):
    # RANDOM_INIT Initialize centroids choosing num_clusters rows of data at random
    #
    #   Input : data         : N-by-D data matrix, where N is the number of data,
    #                          D is the number of dimensions
    #           num_clusters : Number of clusters
    #
    #   Output: centers : the ind of centers,K-by-1 matrix, where K is num_clusters
    centers = np.random.randint(0, data.shape[0], size=num_clusters)
    return centers


# --------------------------------------------------------------------------
def plus_init(core_dist, num_clusters, rho):
    # Optimization of initial cluster centers: first select the object with the maximum density as the first cluster center,
    # then select the object farthest from the first center as the second cluster center, continuing in this manner until K cluster centers are chosen
    centers = np.zeros(num_clusters, dtype=int)
    maxid = int(np.argmax(rho))
    centers[0] = maxid  # Index of the first cluster center
    num_centers = 1
    n = core_dist.shape[1]  # Get number of objects
    while num_centers < num_clusters:
        # Compute distances from all other points to each selected cluster center
        dists = np.zeros((n, num_centers))
        for i in range(num_centers):
            dists[:, i] = core_dist[centers[i], :]
        mindist = np.min(dists, axis=1)
        maxid2 = int(np.argmax(mindist))
        centers[num_centers] = maxid2
        num_centers = num_centers + 1
    return centers


def orth_init(data, num_clusters):
    # ORTH_INIT Initialize orthogonal centers for k-means clustering algorithm.
    #
    #   Input : data         : N-by-D data matrix, where N is the number of data,
    #                          D is the number of dimensions
    #           num_clusters : Number of clusters
    #
    #   Output: init_centers : K-by-D matrix, where K is num_clusters

    #
    # Find the num_clusters centers which are orthogonal to each other
    #
    Uniq = np.unique(data, axis=0)  # Avoid duplicate centers
    num = Uniq.shape[0]
    first = np.random.randint(0, num)  # Randomly select the first center
    init_centers = np.zeros((num_clusters, data.shape[1]))  # Storage for centers
    init_centers[0, :] = Uniq[first, :]
    Uniq = np.delete(Uniq, first, axis=0)
    c = np.zeros(num - 1)  # Accumalated orthogonal values to existing centers for non-centers
    # Find the rest num_clusters-1 centers
    for j in range(1, num_clusters):
        c = c + np.abs(np.dot(Uniq, init_centers[j - 1, :]))
        i = int(np.argmin(c))  # Select the most orthogonal one as next center
        init_centers[j, :] = Uniq[i, :]
        Uniq = np.delete(Uniq, i, axis=0)
        c = np.delete(c, i)
    return init_centers
