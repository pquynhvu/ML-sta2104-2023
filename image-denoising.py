import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as Image
from os.path import exists
from wget import download
from tqdm import tqdm

filename, url = "3vaef0cog4f61.png", "https://i.redd.it/3vaef0cog4f61.png"

def load_img():
    if not exists(filename):
        download(url)

    with open(filename, 'rb') as fp:
        img2 = Image.open(fp).convert('L')
        img2 = np.array(img2)
    return (img2[:96,11:107] > 120) * 2.0 - 1

img_true = load_img()
plt.imshow(img_true, cmap='gray')
plt.show()

# Introduce noise into the image, for each pixel, swap its value between 1 and -1 with rate 0.2.

def gen_noisyimg(img, noise = 0):
    swap = np.random.binomial(n = 1, p = noise, size=img.shape)
    return img * (2 * swap - 1)

noise = 0.2
img_noisy = gen_noisyimg(img_true, noise)
plt.imshow(- 1*img_noisy, cmap='gray')
plt.show()

# Loopy belief propagation (Loopy-BP) algorithm

## Initialization

y = img_noisy.reshape([img_true.size, ])
num_nodes = len(y) # 9216
init_message = np.zeros([2, num_nodes, num_nodes]) + .5
J = 1.0
beta = 1.0
x_i = [- 1.0, 1.0]
x_j = [- 1.0, 1.]

def get_neighbors_of(node):
    neighbors = []
    m = int(np.sqrt(num_nodes))
    if (node + 1) % m != 0:
        neighbors += [node + 1]
    if node % m != 0:
        neighbors += [node - 1]
    if node + m < num_nodes:
        neighbors += [node + m]
    if node - m >= 0:
        neighbors += [node - m]

    return set(neighbors)

## Implement message passing in BP

def get_message(node_from, node_to, messages):
    neighbors_j = list(get_neighbors_of(node_from))
    if node_to in neighbors_j:
       neighbors_j.remove(node_to)

    m_kj = messages[:, neighbors_j, node_from] # extract M_{kj} for both x_j = 1 and x_j = - 1
    prod_m_kj = np.prod(m_kj, axis = 1) # product of inbound messages
    psi_j = np.exp(np.multiply(beta*y[node_from], x_j))  # potential psi_j
    psi_ij = np.array([np.exp(J*x_i[0]*x_j[0]), np.exp(J*x_i[0]*x_j[1]),
                       np.exp(J*x_i[1]*x_j[0]), np.exp(J*x_i[1]*x_j[1])]).reshape(2, 2) # potential psi_ij
    Psi = np.multiply(psi_j, psi_ij).T
    m_marginal = np.multiply(prod_m_kj, Psi)
    m_new= np.sum(m_marginal, axis = 1)

    return m_new
    pass

def step_bp(step, messages):
    for node_from in range(num_nodes):
        for node_to in get_neighbors_of(node_from):
            m_new = get_message(node_from, node_to, messages)
            m_new = m_new / np.sum(m_new) # normalize

            messages[:, node_from, node_to] = step * m_new + (1. - step) * \
                messages[:, node_from, node_to]
    return messages

num_iter = 10
step = 0.5
for it in range(num_iter):
    init_message = step_bp(step, init_message)
    print(it + 1,'/',num_iter)

## Computing belief from messages 
    
def update_beliefs(messages):
    beliefs = np.zeros([2, num_nodes])
    x_i = np.array([- 1, 1])

    for node in range(num_nodes):
          psi_i = np.exp(np.multiply(beta*y[node], x_i))  # potential psi_i
          neighbors_i = list(get_neighbors_of(node))
          m_ji = messages[:, neighbors_i, node]
          prod_m_ji = np.prod(m_ji, axis = 1)
          b_i_tilde = np.multiply(psi_i, prod_m_ji)
          b = b_i_tilde/np.sum(b_i_tilde)
          beliefs[:, node] = b
    return beliefs

beliefs = update_beliefs(init_message)

pred = 2. * ((beliefs[1, :] > .5) + .0) - 1.
img_out = pred.reshape(img_true.shape)

plt.imshow(np.hstack([img_true, -1*img_noisy, -1*img_out]), cmap='gray')
plt.show()

# Momentum in belief propagation

def test_trajectory(step_size, max_step=10):
    # re-initialize each time
    messages = np.zeros([2, num_nodes, num_nodes]) + .5
    images = []
    for t in range(max_step):
        messages = step_bp(step_size, messages)
        beliefs = update_beliefs(messages)
        pred = 2. * ((beliefs[1, :] > .5) + .0) - 1.
        img_out = -1*pred.reshape(img_true.shape)
        images.append(img_out)
    return images

def plot_series(images):

  n = len(images)
  fig, ax = plt.subplots(1, n)
  for i in range(n):
    ax[i].imshow(images[i], cmap='gray')
    ax[i].set_axis_off()
  fig.set_figwidth(10)
  fig.show()

step_sizes = [0.1, 0.3, 1.0]

images_test = []
for t in step_sizes:
    print(t)
    images = test_trajectory(t)
    images_test.append(images)
    plot_series(images)

# Noise level,  beta  and overfitting.
    
## Generate and display images with noise of  0.05,  0.3 
    
noises = [0.05, 0.3]
noised_img = []
for n in noises:
    noised_img_i = gen_noisyimg(img_true, n)
    noised_img.append(-1*noised_img_i)
plot_series(noised_img)

## Perform image denoising on images with noise levels  0.05  and  0.3  using  β=0.5 ,  β=1.0 ,  β=2.5 , and  β=5.0

betas = [0.5, 1, 2.5, 5]
step = 0.8
max_step = 5
images_test = []
for img in noised_img:
    for beta in betas:
        print(beta)
        images = test_trajectory(step, max_step = max_step)
        images_test.append(images)
        plot_series(images)