
python_sources = [
  '__init__.py',
  '_bench_tps.py',
  '_bench_tps_matrix.py',
  '_try_cusolve.py',
  '_try_griddata.py',
  '_try_interpolation.py',
  '_try_tps.py',
  '_try_tps_2.py',
  '_try_tps_class.py',
  '_try_tps_subdom.py',
  'griddata.py',
  'test_thin_plate_spline.py',
  'thin_plate_spline.py',
  'thin_plate_spline_subdom.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidimage/calcul/interpolate'
)

run_command(['transonic', '--meson', '--backend', backend, 'thin_plate_spline.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
