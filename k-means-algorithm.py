import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt

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

# Implement and Run K-Means algorithm

def cost(data, R, Mu):
    D = data.ndim
    N = data.shape[0]
    K = data.shape[1]
    J = 0
    for i in range(N):
      J += sum(((data[i] - Mu.T)**2).sum(axis=1)*R[i])
    return J

## K-Means assignment 

def km_assignment_step(data, Mu):
    D = data.ndim       # dimension of datapoint
    N = data.shape[0]   # Number of datapoints
    K = data.shape[1]   # number of clusters
    r = []
    for i in range(N):
        r.append(np.sqrt(((data[i] - Mu.T)**2).sum(axis=1))) # distances fron each datapoints to each cluster means
    
    r = np.asarray(r) 
    arg_min = np.argmin(r, axis=1) # index where r has the lower value for each obs
    R_new = np.zeros_like(r) # initialize matrix R_new 
    R_new[np.arange(len(R_new)), arg_min] = 1
    return R_new

## K-Means refitting

def km_refitting_step(data, R, Mu):
    D = data.ndim       # dimension of datapoint
    N = data.shape[0]   # Number of datapoints
    K = data.shape[1]   # number of clusters
    Mu_new = []
    for i in range(D):
      Mu_new.append(np.dot(R[:,i], data)/sum(R[:,i]))
    Mu_new = np.asarray(Mu_new).T.reshape(D, K)
    return Mu_new

## Call the K-Means algorithm.

N, D = 400, 2
K = 2
max_iter = 100
class_init = np.random.randint(2, size=N) 
R = np.zeros((N, K))
R[class_init == 1, 1] = 1
R[:,0] = 1 - R[:, 1]

class_1_true_ix = np.asarray(np.where(labels == 0))
class_2_true_ix = np.asarray(np.where(labels == 1))

Mu = np.zeros([D, K])
Mu[:, 1] = 1.
R.T.dot(data), np.sum(R, axis=0)
c = []
for it in range(max_iter):
    R = km_assignment_step(data, Mu)
    Mu = km_refitting_step(data, R, Mu)
    c.append(cost(data, R, Mu))
    print(it, cost(data, R, Mu))
class_1_KM = data[np.where(R[:,0] == 1)]
class_2_KM = data[np.where(R[:,1] == 1)]

## Scatterplot for the data points showing the K-Means cluster assignments of each point.

ix_KM = list(range(100))
fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(class_1_KM[:,0], class_1_KM[:,1], 'x', color='darkorange', label ='First class') 
ax1.plot(class_2_KM[:,0], class_2_KM[:,1], 'o', color='darkorchid', label ='Second class')
ax1.plot(np.array([0.1, 6.0]), np.array([0.1, 0.1]), '^', color = "crimson", label = "True means")
ax1.plot(Mu[0,:], Mu[1,:], 'P', color = "g", label = "Estimated means")
ax1.set_title('K-means cluster assignment')

ax2.plot(ix_KM, c, color='darkorange') 
ax2.set_title('Cost of K-means over 100 iterations')

fig.tight_layout()
plt.show()

new_label = []
for i in range(N):
  if R[i, 0] == 1:
    new_label.append(0)
  elif R[i, 1] == 1:
    new_label.append(1)
new_label = np.asarray(new_label)
print("Misclassification error of K-means is ", np.mean(new_label != labels))