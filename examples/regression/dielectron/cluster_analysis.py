# import numpy
# from sigmoid.preprocessing.datasets import StandardDataset

# dataset = StandardDataset('./temp_switch_cache.h5', cache_all=False, as_tensor=False)
# X_list = []
# for x, _ in dataset:
#     X_list.append(x)
# X = numpy.vstack(X_list)
# print(X.shape)
# numpy.save('codecs.npy', X)

# import sys
# sys.exit(0)


import numpy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
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

# Run the fit
clust = HDBSCAN(
            min_samples=6, 
            min_cluster_size=5,
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.45, 
            n_jobs=24,
            metric='euclidean')
clust.fit(X)


# labels_050 = cluster_optics_dbscan(
#     reachability=clust.reachability_,
#     core_distances=clust.core_distances_,
#     ordering=clust.ordering_,
#     eps=0.01,
# )
# labels_200 = cluster_optics_dbscan(
#     reachability=clust.reachability_,
#     core_distances=clust.core_distances_,
#     ordering=clust.ordering_,
#     eps=0.05,
# )

space = np.arange(len(X))
# reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_# [clust.ordering_]

plt.figure(figsize=(5 * 3, 2 * 5))
G = gridspec.GridSpec(2, 3)
ax11 = plt.subplot(G[0, 0])
ax12 = plt.subplot(G[0, 1])
ax13 = plt.subplot(G[0, 2])

ax21 = plt.subplot(G[1, 0])
ax22 = plt.subplot(G[1, 1])
ax23 = plt.subplot(G[1, 2])

# ax31 = plt.subplot(G[2, 0])
# ax31 = plt.subplot(G[2, 1])
# ax31 = plt.subplot(G[2, 2])

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
# Reachability plot
# colors = ["g.", "r.", "b.", "y.", "c."]
# for klass in numpy.unique(labels):
#     color_idx = klass % len(colors)
#     color = colors[color_idx]
#     Xk = space[labels == klass]
#     Rk = reachability[labels == klass]
#     ax1.plot(Xk, Rk, color, alpha=0.3)
# ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
# ax1.plot(space, np.full_like(space, 0.1, dtype=float), "k-", alpha=0.5)
# ax1.plot(space, np.full_like(space, 0.05, dtype=float), "k-.", alpha=0.5)
# ax1.set_ylabel("Reachability (epsilon distance)")
# ax1.set_title("Reachability Plot")

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

# DBSCAN at 0.5
# colors = ["g.", "r.", "b.", "c."]
# for klass, color in enumerate(colors):
#     Xk = X[labels_050 == klass]
#     ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], "k+", alpha=0.1)
# ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")

# DBSCAN at 2.
# colors = ["g.", "m.", "y.", "c."]
# for klass, color in enumerate(colors):
#     Xk = X[labels_200 == klass]
#     ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], "k+", alpha=0.1)

plt.tight_layout()
plt.savefig("analysis.png")


