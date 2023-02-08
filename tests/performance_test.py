import time

import numpy as np

from src.algorithm_functions import get_potential_splits, get_potential_splits_old

start = time.time()
x = np.random.rand(5_000_000, 5)
get_potential_splits_old(x)
end = time.time()
print(f'Python func: {end - start}')

start = time.time()
x = np.random.rand(5_000_000, 5)
get_potential_splits(x)
end = time.time()
print(f'Numba func: {end - start}')
