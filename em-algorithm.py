import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt
from numpy.core.umath_tests import matrix_multiply as mm

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