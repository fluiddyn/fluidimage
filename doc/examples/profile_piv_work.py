"""Profile PIV work with pyinstrument

Result with fluidimage 0.4.4:

total: ~4.7 s

- ~25% _compute_indices_max, fluidimage/calcul/correl.py:78
- ~25% CorrelFFTW.__call__,  fluidimage/calcul/correl.py:690  (10.4% fftshift, ~1% fft)
- ~10% WorkPIVFromDisplacement._crop_im0 (and 1) fluidimage/works/piv/singlepass.py:403
- ~17% smooth_clean   fluidimage/calcul/smooth_clean.py:25    (9% griddata)

"""

import os

from pyinstrument import Profiler
from pyinstrument.renderers import ConsoleRenderer

from fluidimage.piv import Work

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
params.multipass.threshold_tps = 1.5

params.fix.correl_min = 0.15
params.fix.threshold_diff_neighbour = 3

work = Work(params=params)

profiler = Profiler()
profiler.start()
piv = work.process_1_serie()
profiler.stop()

print(
    profiler.output(
        renderer=ConsoleRenderer(
            unicode=True,
            color=True,
            show_all=False,
            time="percent_of_total",
            # flat=True,  # buggy with pyinstrument 4.6.2!
        )
    )
)

# piv.display(show_interp=False, scale=1, show_error=True)
