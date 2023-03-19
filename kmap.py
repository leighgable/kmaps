
import jax.numpy as jnp
from jax.numpy import logical_and
from jax.numpy import outer
from jax.numpy import argmin
from jax import jit, vmap
from jax import random

from utils.data_loader import load_it

from math import ceil
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

iris = load_it('data', 'iris.pkl')

def normalize(data, axis=1):
    return data/jnp.linalg.norm(data, axis=axis, keepdims=True)
        
tic = time.perf_counter()
iris_data_normal = normalize(iris.data)
toc = time.perf_counter()
print(f"It took jax {toc - tic:0.4f} seconds to normalize the data.")

def kmap(input_shape, sigma=1.0, learning_rate=0.5, seed=None):
    int x = math.ceil(jnp.sqrt(jnp.sqrt(input_shape[0]*5))) # calculates dimension x, y
    int y = x                                               # of meshgrid wrt an 
    activation_map = jnp.zeros((x, y))                      # heuristic (sqrt(sqrt(N)*5))
    
    neighbourx = jnp.arange(x, dtype='float32')  
    neighboury = jnp.arange(y, dtype='float32')      
    _xx, _yy = jnp.meshgrid(neighbourx, neighboury)
    
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    _weights = random.uniform(subkey, shape=(x, y, input_shape[1]), dtype='float32')
    
    
def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1 + t/(max_iter/2))

def _bubble(x, y, c, sigma):
    """Constant function centered in c with spread sigma.
    sigma should be an odd value.
    """
    ax = logical_and(neighbourx > c[0]-sigma,
                     neighbourx < c[0]+sigma)
    ay = logical_and(neighboury > c[1]-sigma,
                     neighboury < c[1]+sigma)
    return outer(ax, ay)*1.

def _euclidean_distance(x, w):
        return jnp.linalg.norm(subtract(x, w), axis=-1)
        
def winner(x):
    """Computes the coordinates of the winning ( i.e. closest ) node."""
    _activate(x)
    return unravel_index(_activation_map.argmin(),
                             _activation_map.shape)


def update(x, win, t, max_iteration):
    """Updates the weights of the nodes.
    Parameters
    ----------
    x : np.array
        Current pattern to learn.
    win : tuple
        Position of the winning node for x (array or tuple).
    t : int
        rate of decay for sigma and learning rate
    max_iteration : int
        If use_epochs is True:
            Number of epochs the SOM will be trained for
        If use_epochs is False:
            Maximum number of iterations (one iteration per sample).
    """
    eta = _decay_function(_learning_rate, t, max_iteration)
    # sigma and learning rate decrease with the same rule
    sig = _decay_function(_sigma, t, max_iteration)
    # improves the performance
    g = neighborhood(win, sig)*eta
    # w_new = eta * neighborhood_function * (x-w)
    _weights += einsum('ij, ijk->ijk', g, x - _weights)        

def pca_weights(data):
     pc_length, pc = jnp.linalg.eig(
                        jnp.cov(jnp.transpose(data)))
        pc_order = argsort(-pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(_neighbourx))):
            for j, c2 in enumerate(linspace(-1, 1, len(_neighboury))):
                _weights[i, j] = c1*pc[:, pc_order[0]] + \
                                      c2*pc[:, pc_order[1]]

def _distance_from_weights(data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight from minisom repo
        """
        input_data = jnp.array(data)
        weights_flat = _weights.reshape(-1, _weights.shape[2])
        input_data_sq = jnp.power(input_data, 2).jnp.sum(axis=1, keepdims=True)
        weights_flat_sq = jnp.power(weights_flat, 2).jnp.sum(axis=1, keepdims=True)
        cross_term = jnp.dot(input_data, weights_flat.T)
        return jnp.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)

kmap = kmap(input_shape=iris.data.shape)
