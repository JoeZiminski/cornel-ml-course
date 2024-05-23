import matplotlib.pyplot as plt
import numpy as np
import keras


PLOT = True

class Perception:

    def __init__(self, max_iter=1000):

        self.max_iter = max_iter
        self.w = None

    def fit(self, X, labels):
        """
        X - first dimension is sample, second dimension is data
        """
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2 and 1 in unique_labels and -1 in unique_labels, "labels must be -1 and 1"

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

        self.w = w

    def predict(self, X):
        """
        Predict label values ({-1, 1}) given the data X.
        """
        N = X.shape[0]
        X = np.c_[X, np.ones(N)]
        y = np.empty(N)

        for i in range(N):
            y[i] = -1 if self.w.T@X[i, :] < 0 else 1

        return y


# Make a random linearly separable dataset by sampling N points randomly and
# then sampling a random plane by which to divide them. Use perceptron to
# fit the plane.
# --------------------------------------------------------------------------------------

w_ = np.random.randint(-10, 10, 3)
b = 0  # does not work with plotting as don't want to add a 4th dim.

w = np.r_[w_, b][:, np.newaxis]

N = 100
X = np.c_[
    np.random.randint(0, 50, N),
    np.random.randint(0, 50, N),
    np.random.randint(-50, 50, N),
    np.ones(N)
].T

labels = np.zeros(N).astype(np.int16)

for i in range(N):
    labels[i] = -1 if w.T@X[:, i] < 0 else 1

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[0, :], X[1, :], X[2, :], c=labels)
ax.set_title("Random labelled points")

perceptron = Perception(max_iter=10000)
perceptron.fit(X[:3, :].T, labels)

w = perceptron.w

if PLOT:
    axis = np.linspace(0, 50, 1000)
    xx, yy = np.meshgrid(axis, axis)

    zz = (-w[0]*xx + -w[1]*yy +w[3])/w[2]
    # plot the plane
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    ax.scatter(X[0, :], X[1, :], X[2, :], c=labels)
    ax.set_title("Fit the hyperplane")
    plt.show()

# Use MNIST numbers dataset to predict if number is 0 or 1. Download dataset,
# use only 0 and 1 fit hyperplane in 28*28 dimensions, use it for predict
# number with great accuracy.
# --------------------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

zero_or_one_train = np.logical_or(y_train == 0, y_train == 1)
x_train = x_train[zero_or_one_train]
y_train = y_train[zero_or_one_train]

y_train = y_train.astype(np.int16)
y_train[y_train == 0] = -1

x_train_reshape = np.reshape(x_train, (y_train.size, 28*28))

perceptron_mnist = Perception(max_iter=1000)
perceptron_mnist.fit(x_train_reshape, y_train)

zero_or_one_test = np.logical_or(y_test == 0, y_test == 1)
x_test = x_test[zero_or_one_test]
y_test = y_test[zero_or_one_test]

x_test_reshape = np.reshape(x_test, (y_test.size, 28*28))

y_test = y_test.astype(np.int16)
y_test[y_test == 0] = -1

y_predict = perceptron_mnist.predict(x_test_reshape)

print("MNIST accuracy: ", np.sum(y_test == y_predict) / y_predict.size)
