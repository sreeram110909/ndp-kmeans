import os
import sys
import numpy as np

# Add python_code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
# pyrefly: ignore[missing-import]
from NDP_Kmeans import NDP_Kmeans

# Ensure results folder exists
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# List of datasets: (filename, number_of_clusters, alpha)
datasets = [
    ('flame.txt', 2, 0.0),
    ('agg.txt', 7, 0.0),
    ('db2.txt', 2, 0.0),
    ('cth.txt', 2, 0.0),
    ('ls.txt', 6, 0.0),
]

for filename, k, alpha in datasets:
    path = os.path.join(os.path.dirname(__file__), 'datasets', filename)
    print(f"\n--- Running {filename} (clusters={k}, alpha={alpha}) ---")
    
    data = np.loadtxt(path)
    X = data[:, :2]
    
    # Save output plot as JPG in results folder with number of clusters in filename
    base_name = os.path.splitext(filename)[0]
    save_img = os.path.join(results_dir, f"{base_name}_clusters_{k}.jpg")
    
    labels = NDP_Kmeans(X, num_clusters=k, alpha=alpha, save_name=save_img)
    print(f"Done. Cluster sizes: {np.bincount(labels)}")
