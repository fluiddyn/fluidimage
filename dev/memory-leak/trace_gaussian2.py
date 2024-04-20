
import gc
import tracemalloc

import numpy as np

from fluidimage.calcul.subpix import compute_subpix_2d_gaussian2

import profiler


n0, n1 = 32, 32

correl = np.zeros((n0, n1), dtype=np.float32)

i_max = n0//2

correl[i_max-1:i_max+2, i_max-1:i_max+2] = 0.6
correl[i_max, i_max] = 1.0
correl[i_max, i_max-1] = 0.7

print(compute_subpix_2d_gaussian2(correl, i_max, i_max))

tracemalloc.start()

for _ in range(5):
    for idx in range(1000):
        compute_subpix_2d_gaussian2(correl, i_max, i_max)
    gc.collect()
    profiler.snapshot()

profiler.display_stats()
profiler.compare()
profiler.print_trace()
