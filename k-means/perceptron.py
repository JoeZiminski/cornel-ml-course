import matplotlib.pyplot as plt
import numpy as np
import keras

class Perception:
    def __init__(self, max_iter=1000):

        self.max_iter = 100
        self.w = None

    def fit(self, X, labels):

        N = X.shape[0]
        orig_dim = X.shape[1]

        w = np.zeros(orig_dim + 1)[:, np.newaxis]
        X_bias = np.c_[X, np.ones(N)]

        iter = 0
        while iter < self.max_iter:
            for i in range(N):

                x = X_bias[i, :][:, np.newaxis]
                y = labels[i]

                if y * w.T @ x <= 0:
                    w = w + y * x

                iter += 1

                if iter >= self.max_iter:
                    break

        return w




w_ = np.random.randint(-10, 10, 3)
b = 0  # does not work with plotting as don't want to add a 4th dim.


w = np.r_[w_, b][:, np.newaxis]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

axis = np.linspace(0, 50, 1000)
xx, yy = np.meshgrid(axis, axis)

zz = (-w[0]*xx + -w[1]*yy +w[3])/w[2]
# plot the plane
# ax.plot_surface(xx, yy, zz, alpha=0.5)


N = 100
X = np.c_[
    np.random.randint(0, 50, N),
    np.random.randint(0, 50, N),
    np.random.randint(-50, 50, N),
    np.ones(N)
].T

labels = np.zeros(N)

for i in range(N):
    labels[i] = -1 if w.T@X[:, i] < 0 else 1

ax.scatter(X[0, :], X[1, :], X[2, :], c=labels)

perceptron = Perception(max_iter=10000)
w = perceptron.fit(X[:3, :].T, labels)

zz = (-w[0]*xx + -w[1]*yy +w[3])/w[2]
# plot the plane
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx, yy, zz, alpha=0.5)
ax.scatter(X[0, :], X[1, :], X[2, :], c=labels)

plt.show()

# TODO: calculate lambda

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

breakpoint()
