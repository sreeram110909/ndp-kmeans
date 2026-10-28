# -------------------------------------------------------------------------
# Aim:
# draw the cluster results of ndpkmeans
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os

def drawcluster_ndpkmeans(A, cluster, ncluster, filename='cluster_result.jpg'):
    n, d = A.shape
    cmap = plt.colormaps['jet'](np.linspace(0, 1, 64))  # cmap = colormap; (64 colors in default MATLAB colormap)
    for i in range(n):
        if cluster[i] > 0:
            ic = int(((cluster[i]) * 64.0) / (ncluster * 1.0))
            ic = min(max(ic - 1, 0), 63)
            x = A[i, 0]
            y = A[i, 1]
            # z = A[i, 2]
            plt.plot(x, y, 'o', markersize=5, markerfacecolor=cmap[ic], markeredgecolor=cmap[ic])
            # hold on;
        else:
            x = A[i, 0]
            y = A[i, 1]
            # z = A[i, 2]
            plt.plot(x, y, 'k.')  # ,'MarkerSize',10);
            # hold on;
    
    if not filename.lower().endswith(('.jpg', '.jpeg')):
        filename = f"{filename}.jpg"
    
    parent_dir = os.path.dirname(os.path.abspath(filename))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    plt.savefig(filename, format='jpeg', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Result saved as {filename}")
