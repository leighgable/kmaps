import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

key = random.PRNGKey(0)
x = random.normal(key, (10,))

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
tic = time.perf_counter()
jnp.dot(x, x.T).block_until_ready() 
toc = time.perf_counter()
print(f"It took jax.numpy {toc - tic:0.4f} seconds.")

x = np.random.normal(size=(size, size)).astype(np.float32)
tic = time.perf_counter()
jnp.dot(x, x.T).block_until_ready() 
toc = time.perf_counter()
print(f"It took numpy {toc - tic:0.4f} seconds.")

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
tic = time.perf_counter()
jnp.dot(x, x.T).block_until_ready()
toc = time.perf_counter()
print(f"It took device_put {toc - tic:0.4f} seconds.")

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
x = random.normal(key, (1000000,))
tic = time.perf_counter()
selu(x).block_until_ready()
toc = time.perf_counter()
print(f"It took without JIT {toc - tic:0.4f} seconds.")

selu_jit = jit(selu)
tic = time.perf_counter()
selu_jit(x).block_until_ready()
toc = time.perf_counter()
print(f"It took with JIT {toc - tic:0.4f} seconds.")

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(f"The derivative of {x_small}: ", derivative_fn(x_small))

def first_finite_differences(f, x):
  eps = 1e-3
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])


print(f"Verify the derivative of {x_small} by finite difference: ",\
     first_finite_differences(sum_logistic, x_small))

tic = time.perf_counter()
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
toc = time.perf_counter()
print(f"It took numpy {toc - tic:0.4f} seconds.")


tic = time.perf_counter()
x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
toc = time.perf_counter()
print(f"It took with jax {toc - tic:0.4f} seconds.")

# plt.plot(x_jnp, y_jnp);
# plt.savefig('figs/test2.png')

# normalizes a matrix jax style.
def norm(X):    
  X = X - X.mean(0)
  return X / X.std(0)

norm_jitted = jit(norm)

X = np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))

tic = time.perf_counter()
norm(X) 
toc = time.perf_counter()
print(f"It took with numpy {toc - tic:0.4f} seconds to normalize the matrix.")


tic = time.perf_counter()
norm_jitted(X)
toc = time.perf_counter()
print(f"It took with jax {toc - tic:0.4f} seconds to normalize the matrix.")

print(f"Some info about X - shape: {X.shape}")
sample = X[np.random.choice(X.shape[0], 2, replace=False), :]
sample_norm = norm_jitted(X)[np.random.choice(X.shape[0], 2, replace=False), :]
print(f"A sampling of X values: {sample}")
print(f"A sampling of X_norm values: {sample_norm}")

