# NDP-Kmeans: K-Means with Natural Density Peaks for Arbitrary-Shaped Clusters

Python implementation of **K-Means Clustering with Natural Density Peaks (NDP-Kmeans)** for discovering complex, arbitrary-shaped clusters and handling noise.

---

## 📌 Features

- **Natural Density Peaks (NDP) Discovery**: Automatically locates density peaks without requiring manual cutoff distance thresholds.
- **Geodesic Distance (GD) Matrix**: Constructs sparse/shortest path graphs across NDPs to connect non-spherical manifolds and complex shapes.
- **K-Means with GD**: Optimizes cluster centers using geodesic distance representations to accurately segment manifold topologies.
- **Noise Robustness**: Incorporates an $\alpha$ noise proportion parameter for filtering sparse boundary outliers.
- **Visual Clustering Plots**: Generates cluster visualizations saved directly to the `results/` folder.

---

## 📂 Project Structure

```text
├── code/
│   ├── __init__.py
│   ├── NDP_Kmeans.py              # Main clustering pipeline
│   ├── NDP_Searching.py           # Natural Density Peaks search
│   ├── KmeansWithGD.py            # K-means clustering using geodesic distances
│   └── drawcluster_ndpkmeans.py   # Visualization utility
├── datasets/                      # Benchmark 2D clustering datasets (flame, agg, db2, cth, ls, etc.)
├── results/                       # Clustered plot outputs (.jpg)
├── main.py                        # Benchmark runner script across datasets
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sreeram110909/ndp-kmeans.git
   cd ndp-kmeans
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

Run the benchmark script:
```bash
python main.py
```

To run on your own 2D/multi-dimensional dataset:
```python
import numpy as np
from code.NDP_Kmeans import NDP_Kmeans

# Load coordinates (n x d)
X = np.loadtxt("datasets/flame.txt")[:, :2]

# Run NDP-Kmeans (data, num_clusters, alpha_noise_ratio, save_plot_path)
labels = NDP_Kmeans(X, num_clusters=2, alpha=0.0, save_name="results/flame_clusters.jpg")
```

---

## 📊 Benchmark Datasets Included

| Dataset | Clusters ($k$) | Description |
|---|---|---|
| `flame.txt` | 2 | Flame shape dataset |
| `agg.txt` | 7 | Aggregation clustering dataset |
| `db2.txt` | 2 | Non-convex shapes |
| `cth.txt` | 2 | Complex curved manifolds |
| `ls.txt` | 2 | Line and multi-density cluster manifolds |

All output plots are generated in the `results/` directory upon running `main.py`.
