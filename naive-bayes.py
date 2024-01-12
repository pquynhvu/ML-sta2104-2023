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

# Randomly sample and plot 10 images from the learned distribution using the MAP estimates. 

def image_sampler(theta, pi, num_images):
    ## get nuber of features
    num_features = theta.shape[0]
    ## sample from the multinomial class distribution defined by the pi vector a total of num_images times.
    classes_sampled = np.argmax(np.random.multinomial(1, pi, size=num_images), axis = 0)
    ## Initialize list of pixel values for all pixels and all images sampled
    x_j_ls = []

    ## getting pixel values for each image to generate using the sampled classes
    for n in classes_sampled:
        ## getting the MAP theta vector of all pixels for the given sampled class n, resulting in a N_features vector of theta's
        theta_j = theta[:, n]
        my_generator = np.random.default_rng()
        ## sampling all N_features pixel values based on their respective theta's using a binomial generator
        x_j = my_generator.binomial(1, theta_j)
        ## appending the pixel values
        x_j_ls.append(x_j)
    ## converting pixel values for all generate images into an N_imagex x N_features array
    return np.asarray(x_j_ls)

def plot_images(images, ims_per_row=5, padding=5, image_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=0., vmax=1.):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)

    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = vmin
    concat_images = np.full(((image_dimensions[0] + padding) * N_rows + padding,
                             (image_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], image_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + image_dimensions[0]) * row_ix
        col_start = padding + (padding + image_dimensions[1]) * col_ix
        concat_images[row_start: row_start + image_dimensions[0],
                      col_start: col_start + image_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    plt.plot()
    plt.show()

sampled_images = image_sampler(theta_est, pi_est, 10)
plot_images(sampled_images)

# We assume that only 30% of the pixels are observed. For the first 20 images in the training set, plot the images 
# when the unobserved pixels are left as white, as well as the same images when the unobserved pixels are filled with 
# the marginal probability of the pixel being 1 given the observed pixels

def probabilistic_imputer(theta, pi, original_images, is_observed):

    ## initializing list of impute images' pixel values
    imputed_image1_ar = []
    ## getting the images' pixel values, post-imputation, for each image one at a time
    for i in range((original_images).shape[0]):
        ## taking image i and reshaping into a 1xN_features array
        original_image1 = train_images[i].reshape(1, train_images.shape[1])
        ## taking the is_observed matrix for image i and reshaping into a 1xN_features array
        is_observed1 = is_observed[i, :].reshape(1, is_observed.shape[1])
        ## calculating the Bernoulli pdf output for each pixel
        ### term_observed_1 is the first part of the bernoulli concerning the probability of pixel a being 1
        term_observed_1 = np.power(theta, original_image1.T)
        ### term_observed_2 is the second part of the bernoulli concerning the probability of pixel a being 0
        term_observed_2 = np.power((1-theta), (1-original_image1).T)
        ### multiplying the probabilities element-wise to get the Bernoulli pdf output
        term_observed_prod = np.multiply(term_observed_1, term_observed_2)
        ## filtering the Bernoulli pdf outputs to only include those that are actually observed
        term_observed_prod_filtered = term_observed_prod[(is_observed1 == 1).T[:, 0]]
        ## multiplying the bernoulli pdf outputs of all the observed pixels to get their likelihood
        term_observed_prod_filtered_prodded = np.prod(term_observed_prod_filtered, axis = 0)
        ## multiplying the observed pixels' likelihood by pi to get a constant N_classes vector that is present in both the numerator and denominator of the probability of unobserved pixel j given observed pixels E.
        constant_k = pi * term_observed_prod_filtered_prodded
        ## calcuating the denominator of the desired probability by summing up the per-class constant vector across all vectors, giving a scalar value
        denominator = np.sum(constant_k)
        ## calculating the bernoulli pdf output of the unobserved pixel j for all pixels separately
        bernoulli_jk = np.multiply(np.power(theta, original_image1.T), np.power((1-theta), (1-original_image1).T))

        probabilities_j = (np.sum((bernoulli_jk*constant_k), axis = 1))/denominator
        imputed_image1 = original_image1
        # imputed_image1[(is_observed1 == 0)]
        imputed_image1[0, (is_observed1 == 1).T[:, 0]] = probabilities_j[(is_observed1 == 1).T[:, 0]]

        imputed_image1_ar.append(imputed_image1)
    imputed_images = np.array(imputed_image1_ar).reshape(original_images.shape[0], is_observed.shape[1])
    return imputed_images

num_features = train_images.shape[1]
is_observed = np.random.binomial(1, p=0.3, size=(20, num_features))
plot_images(train_images[:20] * is_observed)

imputed_images = probabilistic_imputer(theta_est, pi_est, train_images[:20], is_observed)
plot_images(imputed_images)