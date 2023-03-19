from functools import partial
import jax.numpy as jnp
from jax import random


key = random.PRNGKey(0)
x = random.randint(key, (10,2), 0, 10)
target = random.randint(key, (2,), 0, 10)

def euclidean(v1, v2):
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    return jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2)

partial_distance = partial(euclidean, target) 

x_sort = sorted(x, key=partial_distance)

for p in x:
    # print(euclidean(p, target))
    print(f"Unsorted: {partial_distance(p)}")
    
for p in x_sort:
    print(f"Sorted: {partial_distance(p)}")