import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class KNN:
    """

    X : (n_samples, n_features)
    """
    def __init__(self, k=5):
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        N = X.shape[0]
        labels = np.zeros(N)
        for i in range(N):
            dists = np.sum((X[i, :] - self.X)**2, axis=1)

            nearest_idx = np.argsort(dists)[:self.k]

            labels[i] = stats.mode(self.y[nearest_idx]).mode

        return labels

# Sample some expo r.v.
X_train = np.vstack([
    np.random.exponential(5, size=(100, 2)) + np.c_[np.ones(100) * 20, np.ones(100) * 10],
    np.random.exponential(5, size=(100, 2)) + np.c_[np.ones(100) * 10, np.ones(100) * 20],
])
Y_train = np.r_[np.zeros(100), np.ones(100)]

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
plt.title("Training data")
plt.show()

# Save train data, predict
knn = KNN(1)

knn.fit(X_train, Y_train)

labels = knn.predict(X_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=labels)
plt.title("Predict the training data")
plt.show()

# sample some new data, predict that
X_new = np.vstack([
    np.random.exponential(5, size=(100, 2)) + np.c_[np.ones(100) * 20, np.ones(100) * 10],
    np.random.exponential(5, size=(100, 2)) + np.c_[np.ones(100) * 10, np.ones(100) * 20],
])

plt.scatter(X_new[:, 0], X_new[:, 1], c=Y_train)
plt.title("New data")
plt.show()

labels = knn.predict(X_new)

plt.scatter(X_new[:, 0], X_new[:, 1], c=labels)
plt.title("Predict new data")
plt.show()

# Show the decision boundary
res = 200

xs, ys = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 50, res))

labels = knn.predict(np.c_[xs.flatten(), ys.flatten()])
boundary = np.reshape(labels, (res, res))

plt.imshow(boundary, origin="lower")
plt.show()

