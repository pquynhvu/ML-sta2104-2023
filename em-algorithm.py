import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt
from numpy import matmul as mm

# Generating the Data
# Set the number of points N=400, their dimension  D=2, and the number of clusters  K=2 , Sample 200 data points for k=1 and 200 for k=2

num_samples = 400
num_k = 200
cov = [[10, 7], [7, 10]]
mean_1 = [0.1, 0.1]
mean_2 = [6.0, 0.1]

x_class1 = np.random.multivariate_normal(mean_1, cov, num_k)
x_class2 = np.random.multivariate_normal(mean_2, cov, num_k)
xy_class1 = np.insert(x_class1, 2, np.full(num_k, 0), axis=1) # 0 for class 1
xy_class2 = np.insert(x_class2, 2, np.full(num_k, 1), axis=1) # 1 for class 2

data_full = np.stack((xy_class1, xy_class2)).reshape(num_samples, 3)
np.random.shuffle(data_full)
data = data_full[:, 0:2]
labels = data_full[:,2]

# Make a scatter plot of the data points showing the true cluster assignment of each point using different color codes and shape (x for first class and circles for second class)

plt.plot(x_class1[:,0], x_class1[:,1], 'x', color='darkorange', label ='First class') 
plt.plot(x_class2[:,0], x_class2[:,1], 'o', color='darkorchid', label ='Second class')
plt.plot(np.array([0.1, 6.0]), np.array([0.1, 0.1]), '^', color = "crimson", label = "True means")
plt.legend()
plt.show()

#  Implement EM algorithm for Gaussian mixtures

def normal_density(x, mu, Sigma):
    return np.exp(-.5 * np.dot(x - mu, np.linalg.solve(Sigma, x - mu))) \
        / np.sqrt(np.linalg.det(2 * np.pi * Sigma))

def log_likelihood(data, Mu, Sigma, Pi):
    N = data.shape[0]   # Number of datapoints
    D = data.ndim       # dimension of datapoint
    K = Mu.shape[1]     # number of mixtures
    L, T = 0., 0. 
    for n in range(N):
      sum = 0
      for k in range(K):
        sum += Pi[k]*normal_density(data[n,:], Mu[:,k], Sigma[k])
      L += np.log(sum)
    return L

def gm_e_step(data, Mu, Sigma, Pi):
    N = data.shape[0]   # Number of datapoints
    D = data.ndim       # dimension of datapoint
    K = Mu.shape[1]     # number of mixtures
    Gamma = np.zeros((N, K))
    for n in range(N):
      for k in range(K):
        Gamma[n, k] = Pi[k]*normal_density(data[n,:], Mu[:,k], Sigma[k])
    Gamma = Gamma/Gamma.sum(axis=1)[:, None]
    return Gamma

def gm_m_step(data, Gamma):
    N = data.shape[0]   # Number of datapoints
    D = data.ndim       # dimension of datapoint
    K = Gamma.shape[1]  # number of mixtures
    Nk = Gamma.sum(axis = 0)
    Mu = np.dot(data.T, Gamma)/Nk
    Pi = Nk/N
    
    Sigma = np.zeros((K, D, D))
    for k in range(K):
      y = data - Mu[:,k]
      Sigma[k] = (Gamma[:,k,None,None]*mm(y[:,:,None], y[:,None,:])).sum(axis = 0)

    Sigma /= Gamma.sum(axis=0)[:,None,None]
    return Mu, Sigma, Pi

def gm_m_step(data, Gamma):
    N, D = data.shape 
    K = Gamma.shape[1]  
    Nk = np.sum(Gamma, axis = 0) 
    Mu = np.asarray([np.sum(data * (Gamma[:, k] / Nk[k])[:, np.newaxis], axis = 0) for k in range(K)]).T
    Sigma = [0]*K # TODO
    for k in range(K):
        Sigma[k] = np.sum((Gamma[:, k]/Nk[k])[:, np.newaxis, np.newaxis] * ((data - Mu[:, k])[:,:, None]*(data - Mu[:, k])[:,None, :]), axis = 0)
    
    Pi = Nk/N # TODO
    return Mu, Sigma, Pi

N = data.shape[0]   # number of datapoints
D = data.ndim       # dimension of datapoint
K = data.shape[1]   # number of mixtures
Mu = np.zeros([D, K])
Mu[:, 1] = 1.
Sigma = [np.eye(2), np.eye(2)]
Pi = np.ones(K) / K
Gamma = np.zeros([N, K]) # Gamma is the matrix of responsibilities 
max_iter  = 200
mus  = []
Sigmas = []
Pis = []
for it in range(max_iter):
    Gamma = gm_e_step(data, Mu, Sigma, Pi)
    Mu, Sigma, Pi = gm_m_step(data, Gamma)
    mus.append(Mu)
    Sigmas.append(Sigma)
    Pis.append(Pi)

arg_max = np.argmax(Gamma, axis=1)
class_1_EM = data[np.where(arg_max == 0)]
class_2_EM = data[np.where(arg_max == 1)]

print("Misclassification error of EM is ", np.mean(arg_max!=labels))

log_lik = []
for i in range(200):
  log_lik.append(log_likelihood(data, mus[i], Sigmas[i], Pis[i]))
ix_EM = list(range(200))

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(class_1_EM[:,0], class_1_EM[:,1], 'x', color='darkorange', label ='First class') 
ax1.plot(class_2_EM[:,0], class_2_EM[:,1], 'o', color='darkorchid', label ='Second class')
ax1.plot(np.array([0.1, 6.0]), np.array([0.1, 0.1]), '^', color = "crimson", label = "True means")
ax1.plot(Mu[0,:], Mu[1,:], 'P', color = "g", label = "Estimated means")
ax1.set_title('EM-algorithm cluster assignment')

ax2.plot(ix_EM, log_lik, color='darkorange') 
ax2.set_title('Log-likelihood of EM over 200 iterations')

fig.tight_layout()
plt.show()

# K-means might do better if we add more clusters, as the more clusters we add, the easier it is for the algorithm to reduce 
# the distance between points and centroids, reducing the within variability while the variability between clusters is likely 
# to increase. In EM-algorithm, as we increase the number of clusters, the number of local maxima grows quickly and thus a local 
# maximizer obtained by the EM algorithm can be a very bad estimator that is far away from the MLE.

