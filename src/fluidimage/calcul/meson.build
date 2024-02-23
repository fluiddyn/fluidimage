
python_sources = [
  '__init__.py',
  '_evaluate_subpix.py',
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

run_command(['transonic', '--meson', '--backend', backend, 'correl.py', 'subpix.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
