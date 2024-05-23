import matplotlib.pyplot as plt
import numpy as np


class KMeans3D:
    """
    The k-means algorithm in 3D with 3 labels.
    This is an unsupervised ML method that
    learns a set of k cluster centers with
    minimises the within-cluster variance
    of the dataset.

    https://en.wikipedia.org/wiki/K-means_clustering

    X: (n_samples, n_features)
    """
    def __init__(self, max_iter=300, tol=0.0001):

        self.max_iter=max_iter
        self.tol = tol
        self.cluster_centers = None

    def fit(self, X):
        """
        Given a set of datapoints X find the
        cluster centers that minimises the
        within-cluster variation.
        """
        N = X.shape[0]
        n_dim = X.shape[1]

        cluster_centers = np.eye(n_dim)

        labels = np.zeros(N, dtype=np.int64)

        for iter in range(self.max_iter):

            # Update the label for every datapoint in X
            # to be the closest cluster center
            self.calculate_dist(X, N, cluster_centers, labels)

            # Recalculate cluster centers based on
            # new labels
            cluster_centers_new = np.vstack([
                np.mean(X[labels == 0, :], axis=0),
                np.mean(X[labels == 1, :], axis=0),
                np.mean(X[labels == 2, :], axis=0)
            ]).T

            if np.linalg.norm(cluster_centers - cluster_centers_new, ord="fro") < self.tol:
                print(f"Converge on iteration {iter}")
                break

            cluster_centers = cluster_centers_new

        else:
            print(f"Converged at max_iter: {self.max_iter}")

        self.cluster_centers = cluster_centers

    def predict(self, X):
        """
        Given a set of datapoints X classify
        them based on the Euclidean distance
        with the learnt cluster centers.
        """
        if self.cluster_centers is None:
            raise RuntimeError("Run fitting first.")

        N = X.shape[0]
        labels = np.zeros(N)

        self.calculate_dist(X, N, self.cluster_centers, labels)

        return labels

    def squared_dist(self, dat, centers):
        return np.sum((dat - centers)**2)

    def calculate_dist(self, X, N, cluster_centers, k_ids):
        """
        For each x ∈ X find the cluster center with
        minimum euclidean distance and assign as label.
        """
        for i in range(N):
            dat = X[i, :]
            min_dist = np.argmin(
                [
                    self.squared_dist(dat, cluster_centers[:, 0]),
                    self.squared_dist(dat, cluster_centers[:, 1]),
                    self.squared_dist(dat, cluster_centers[:, 2])
                ]
            )
            k_ids[i] = min_dist

    def plot_scatter_3d(self, X, labels, title):
        """"""
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for id_ in [0, 1, 2]:
            xs = X[labels == id_, 0]
            ys = X[labels == id_, 1]
            zs = X[labels == id_, 2]
            ax.scatter(xs, ys, zs)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.title(title)
        plt.show()


# Run an example
np.random.seed(41)

# Randomly sample from 3 Gaussian distributions
mu = {
    0: np.array([5, 0, 0]),
    1: np.array([0, 0, 0]),
    2: np.array([0, 5, 0])
}
sig = {
    0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    1: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    2: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
}
n_sample = 1000
X = np.hstack([np.random.multivariate_normal(mu[idx], sig[idx], n_sample).T for idx in range(3)]).T
known_labels = np.hstack([np.ones(n_sample) * idx for idx in range(3)])

k_means = KMeans3D()

k_means.plot_scatter_3d(X, known_labels, "Data before labelling")

k_means = KMeans3D()
k_means.fit(X)

new_n_sample = 100
X_new = np.hstack([np.random.multivariate_normal(mu[idx], sig[idx], 100).T for idx in range(3)]).T

predict_orig_labels = k_means.predict(X)
k_means.plot_scatter_3d(X, predict_orig_labels, "Result of training")

predict_new_labels = k_means.predict(X_new)
k_means.plot_scatter_3d(X_new, predict_new_labels, "Result of predict")
