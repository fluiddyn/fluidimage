import gc
import os
import tracemalloc

from fluidimage.piv import Work

import profiler

os.environ["OMP_NUM_THREADS"] = "1"

params = Work.create_default_params()

params.series.path = "../../image_samples/wake_legi/images/B*.png"

params.piv0.shape_crop_im0 = 40
params.piv0.displacement_max = 14

params.piv0.nb_peaks_to_search = 1
params.piv0.particle_radius = 3

params.mask.strcrop = ":, :1500"

params.multipass.number = 2

# params.multipass.use_tps = "last"
params.multipass.use_tps = False
params.multipass.subdom_size = 200
params.multipass.smoothing_coef = 10.0
params.multipass.threshold_tps = 0.5

params.fix.correl_min = 0.15
params.fix.threshold_diff_neighbour = 3

work = Work(params=params)

# tracemalloc.start()

# piv = work.process_1_serie()

# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics("lineno")

# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)

tracemalloc.start(10)

for _ in range(5):
    piv = work.process_1_serie()
    gc.collect()
    profiler.snapshot()

profiler.display_stats()
profiler.compare()
profiler.print_trace()
