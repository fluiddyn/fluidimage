
python_sources = [
  '__init__.py',
  '_evaluate_subpix.py',
  '_try_correl_images_diff_sizes.py',
  '_try_correl_images_same_size.py',
  'correl.py',
  'correl_pycuda.py',
  'errors.py',
  'fft.py',
  'smooth_clean.py',
  'subpix.py',
  'test_correl.py',
  'test_fft.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidimage/calcul'
)

subdir('interpolate')
subdir('plot_evaluate_subpix')
