
python_sources = [
  '__init__.py',
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
