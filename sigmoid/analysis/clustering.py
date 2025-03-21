import numpy
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from hdbscan import HDBSCAN
# from sklearn.cluster import HDBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix

import networkx as nx

class ClusterFinder:

    def __init__(self, **kwargs):

        self.n_comp_ = 3
        self.pca_ = PCA(n_components=self.n_comp_, whiten=True)
        self.scores_ = []
        self.hdbscan_ = HDBSCAN(**kwargs)
        self.aff_prop_ = AffinityPropagation(verbose=True)
        self.min_ = 0
        self.max_ = 0
        '''
        --------------------------------
        arguments for hdbscan
        --------------------------------
        min_samples=20,
        min_cluster_size=50,
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.45,
        n_jobs=24,
        metric='euclidean',
        store_centers='medoid'
        '''

    def find_clusters(self, data: numpy.ndarray) -> numpy.ndarray:
        """ Runs HDSCAN on provided data.

            :param data (numpy.ndarray)
                2D array with data.
        """
        self.pca_.fit(data)
        print("variances")
        for c in range(0, self.n_comp_):
            print(f"[INFO][clustering::PCA] component {c} explains: {self.pca_.explained_variance_ratio_[c]:1.4f} of variance.")
        X = cosine_similarity(data)
        print("[INFO] distances done.")
        # apply cut-ff
        print("[INFO] building sparse matrix.")
        X[X < 0.9] = 0
        X = csr_matrix(X)
        print("[INFO] sparse matrix done.")
        # labels = self.aff_prop_.fit_predict(data)
        print("[INFO] greating graph...")
        # G = nx.from_numpy_array(X)
        G = nx.from_scipy_sparse_array(X)
        G.edges(data=True)
        print("[INFO] Done greating graph.")
        
        print("[INFO] Detecting communities...")
        comu = nx.community.louvain_communities(G, seed=55, backend='cugraph')
        print(f"[INFO] Louvain community detection yielded {len(comu)} communities.")
        
        labels = numpy.zeros(data.shape[0], dtype=numpy.int32)
        for i, c in enumerate(comu):
            H = G.subgraph(c)
            h = list(H.nodes)
            labels[h] = i
        # labels = self.hdbscan_.fit_predict(X)
        return labels

    def elbow_kmeans(self, data, maxK=30, seed_centroids=None):
        """
            parameters:
            - data: pandas DataFrame (data to be fitted)
            - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
            - seed_centroids (default = None ): float (initial value of centroids for k-means)
        """
        sse = {}
        for k in list(range(1, maxK)):
            print("k: ", k)
            kmeans = KMeans(
                n_clusters=k,
                init='random',
                n_init=100,
                algorithm='elkan',
                random_state=0).fit(data)
                # data["clusters"] = kmeans.labels_
            # Inertia: Sum of distances of samples to their closest cluster center
            sse[k] = kmeans.inertia_
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.savefig("elbow.png")

    def plot_clusters(self,
                      data: numpy.ndarray, labels: numpy.ndarray,
                      prefix: str = None):
        """ Plot clusters in 2-D space.
        """
        X = self.pca_.transform(data)
        self.min_ = numpy.min(X) / 0.9
        self.max_ = numpy.max(X) / 0.9

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
                colors = ["g", "r", "b", "y", "c"]
                for klass in numpy.unique(labels):
                    if klass == -1:
                        continue
                    color_idx = klass % len(colors)
                    color = colors[color_idx]
                    Xk = X[labels == klass]
                    points = numpy.asarray([Xk[:, x_dim], Xk[:, y_dim]]).T
                    hull = ConvexHull(points)
                    # for simplex in hull.simplices:
                    #    ax.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=0.5, alpha=0.1)
                    ax.scatter(Xk[:, x_dim], Xk[:, y_dim], s=0.1, c=color, alpha=0.1)
            else:
                ax.plot(X[labels == -1, x_dim], X[labels == -1, y_dim], "k+", alpha=0.01)
            ax.set_xlim(self.min_, self.max_)
            ax.set_ylim(self.min_, self.max_)
            if is_clustered:
                ax.set_title(f"proj = {plot_info['proj']}")
            else:
                ax.set_title(f"proj = {plot_info['proj']} (noise)")

        plt.tight_layout()
        if prefix is not None:
            plt.savefig(f"{prefix}_analysis.pdf")
        else:
            plt.savefig("analysis.pdf")
