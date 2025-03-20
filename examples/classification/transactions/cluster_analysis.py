from tqdm import tqdm
import numpy
from sigmoid.preprocessing.datasets import StandardDataset

# dataset = StandardDataset('./temp_switch_cache.h5', cache_all=False, as_tensor=False)
# X_list = []
# for x, _ in tqdm(dataset):
#     X_list.append(x)
# X = numpy.vstack(X_list)
# print(X.shape)
# numpy.save('codecs.npy', X)

# import sys
# sys.exit(0)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial import ConvexHull
# Generate sample data

np.random.seed(0)
# n_points_per_cluster = 250
# C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
# C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
# C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
# C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# X = np.vstack((C1, C2, C3, C4, C5, C6))

# read data
X = numpy.load('codecs.npy')#[0:90000]

# reduce dims
n_comp = 3
pca = PCA(n_components=n_comp, whiten=True)
X = pca.fit_transform(X)
print("variances")
for c in range(0, n_comp):
    print(f"component {c}: {pca.explained_variance_ratio_[c]:1.4f}")

plt_min = numpy.min(X) / 0.9
plt_max = numpy.max(X) / 0.9

# compute distances
D = pairwise_distances(X, metric='euclidean')
# normalize distances
D = D / numpy.max(D)
run_id = 0

ms = [5, 20]
mcs = [20, 50]
seps = [0.0, 0.5]

for _ms in ms:
    for _mcs in mcs:
        for _seps in seps:
            run_id = run_id + 1
            print(f"run_id = {run_id}")
            # Run the fit
            clust = HDBSCAN(
                        min_samples=_ms, 
                        min_cluster_size=_mcs,
                        cluster_selection_method='eom',
                        cluster_selection_epsilon=_seps,
                        n_jobs=24,
                        metric='precomputed')
            clust.fit(D)

            labels = clust.labels_

            plt.figure(figsize=(5 * 3, 2 * 5))
            G = gridspec.GridSpec(2, 3)
            ax11 = plt.subplot(G[0, 0])
            ax12 = plt.subplot(G[0, 1])
            ax13 = plt.subplot(G[0, 2])

            ax21 = plt.subplot(G[1, 0])
            ax22 = plt.subplot(G[1, 1])
            ax23 = plt.subplot(G[1, 2])

            info = [
                {
                'axis': ax11,
                'x_dim': 0,
                'y_dim': 1,
                'clustered': True,
                'proj': 'xy'
                },
                {
                'axis': ax12,
                'x_dim': 0,
                'y_dim': 2,
                'clustered': True,
                'proj': 'xz'
                },
                {
                'axis': ax13,
                'x_dim': 1,
                'y_dim': 2,
                'clustered': True,
                'proj': 'yz'
                },
                {
                'axis': ax21,
                'x_dim': 0,
                'y_dim': 1,
                'clustered': False,
                'proj': 'xy'
                },
                {
                'axis': ax22,
                'x_dim': 0,
                'y_dim': 2,
                'clustered': False,
                'proj': 'xz'
                },
                {
                'axis': ax23,
                'x_dim': 1,
                'y_dim': 2,
                'clustered': False,
                'proj': 'yz'
                },
            ]

            for plot_info in info:
                ax = plot_info['axis']
                ax.set_aspect('equal')
                x_dim = plot_info['x_dim']
                y_dim = plot_info['y_dim']
                is_clustered = plot_info['clustered']
                
                if is_clustered:
                    colors = ["g.", "r.", "b.", "y.", "c."]
                    for klass in numpy.unique(labels):
                        color_idx = klass % len(colors)
                        color = colors[color_idx]
                        Xk = X[clust.labels_ == klass]
                        points = numpy.asarray([Xk[:, x_dim], Xk[:, y_dim]]).T
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            ax.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=0.5, alpha=0.7)
                        ax.plot(Xk[:, x_dim], Xk[:, y_dim], color, alpha=0.1)
                else: 
                    ax.plot(X[clust.labels_ == -1, x_dim], X[clust.labels_ == -1, y_dim], "k+", alpha=0.01)
                ax.set_xlim(plt_min, plt_max) 
                ax.set_ylim(plt_min, plt_max)
                if is_clustered: 
                    ax.set_title(f"proj = {plot_info['proj']}")
                else:
                    ax.set_title(f"proj = {plot_info['proj']} (noise)")

            plt.tight_layout()
            plt.savefig(f"{run_id:04d}_analysis.png")


