import numpy as np
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)

def fashion_mnist():
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')
    # Remove the data point that cause log(0)
    remove = (14926, 20348, 36487, 45128, 50945, 51163, 55023)
    train_images = np.delete(train_images,remove, axis=0)
    train_labels = np.delete(train_labels, remove, axis=0)
    return train_images, train_labels, test_images[:1000], test_labels[:1000]

def load_fashion_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels =  fashion_mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels

def train_map_estimator(train_images, train_labels):
    x_ab = train_images.T @ train_labels # summing all pixels a of samples
    n_b = np.sum(train_labels, axis = 0) # number of items in each class
    theta_ab = np.multiply((1+x_ab), (1/(n_b+2))) # MAP theta
    pi_b = n_b / np.sum(n_b) # MAP pi
    return theta_ab, pi_b

def log_likelihood(images, theta, pi):
    """ Inputs: images (N_samples x N_features), theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
        log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
        log_like is a matrix of (N_samples x N_classes)
    Note that log likelihood is not only for c^(i), it is for all possible c's."""

    log_like_ar = []
    for i in range(images.shape[0]):
        images_sample1 = images[i, :]
        images_sample1_rs = images_sample1.reshape(1, images_sample1.shape[0])
        x_j_1 = 1 - images_sample1
        theta_1 = 1-theta
        term_1 = np.log(pi).reshape(1, pi.shape[0])
        term_2_1 = images_sample1_rs @np.log(theta)
        term_2_2 = (1-images_sample1_rs) @ np.log(1-theta)
        term_3_unprodded_unsummed_unlogged = np.multiply(np.power(theta, images_sample1_rs.T), np.power((1-theta), (1-images_sample1_rs).T))
        term_3 = np.log(np.sum(np.prod(term_3_unprodded_unsummed_unlogged, axis = 0)*pi))
        log_like_i = term_1 + term_2_1 + term_2_2 - term_3
        log_like_ar.append(log_like_i)
    log_like = np.array(log_like_ar).reshape(images.shape[0], len(pi))
    return log_like

def accuracy(log_like, labels):
    class_hat = np.argmin(log_like, axis = 1)
    class_true = np.argmax(labels, axis = 1)
    accuracy = np.mean(np.equal(class_hat, class_true))
    return accuracy

N_data, train_images, train_labels, test_images, test_labels = load_fashion_mnist()
theta_est, pi_est = train_map_estimator(train_images, train_labels)

loglike_train = log_likelihood(train_images, theta_est, pi_est)
avg_loglike = np.sum(loglike_train * train_labels) / N_data
train_accuracy = accuracy(loglike_train, train_labels)
loglike_test = log_likelihood(test_images, theta_est, pi_est)
test_accuracy = accuracy(loglike_test, test_labels)

print(f"Average log-likelihood for MAP is {avg_loglike:.3f}")
print(f"Training accuracy for MAP is {train_accuracy:.3f}")
print(f"Test accuracy for MAP is {test_accuracy:.3f}")