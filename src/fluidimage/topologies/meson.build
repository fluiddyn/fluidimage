
python_sources = [
  '__init__.py',
  'base.py',
  'bos.py',
  'example.py',
  'image2image.py',
  'launcher.py',
  'log.py',
  'nb_cpu_cores.py',
  'optical_flow.py',
  'piv.py',
  'preproc.py',
  'splitters.py',
  'surface_tracking.py',
  'test_bos.py',
  'test_example.py',
  'test_image2image.py',
  'test_optical_flow.py',
  'test_piv.py',
  'test_preproc.py',
  'test_splitters.py',
  'test_surftracking.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidimage/topologies'
)

run_command(['transonic', '--meson', '--backend', backend, 'example.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
