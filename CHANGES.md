# Release notes

See also the
[unreleased changes](https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.3.0...branch%2Fdefault).

## [0.3.0] (2024-03-05)

```{warning}

This version contains incompatible API changes documented here.

```

### Removed

- The parameter `strcouple` is replaced by `str_subset`.
- `from fluidimage.preproc import PreprocBase` has to be replaced by
  `from fluidimage.preproc import Work`.

### Changed

- Better default for `params.series.str_subset` (`"pairs"` for PIV),
  `params.series.ind_start` (`"first"`), `params.preproc.series.str_subset` (`"all1by1"`)
  and `params.preproc.series.ind_first` (`"first"`).

### Added

- Module {mod}`fluidimage.piv` to import the PIV classes `Work` and `Topology`.

- Module {mod}`fluidimage.image2image` to import classes `Work` and `Topology`
  for user-defined preprocessing.

- Modules {mod}`fluidimage.bos` and {mod}`fluidimage.optical_flow` to import the
  corresponding `Work` and `Topology` classes.

- The work classes {class}`fluidimage.piv.Work`, {class}`fluidimage.preproc.Work` and
  {class}`fluidimage.optical_flow.Work`
  now have parameters in `params.series` and a method
  {func}`fluidimage.works.BaseWorkFromSerie.process_1_serie` (see the examples on
  [preprocessing](./examples/preproc.md) and [PIV](./examples/piv_try_params.md)).

- The work classes {class}`fluidimage.image2image.Work` and
  {class}`fluidimage.bos.Work`
  now have a parameters in `params.images` and a method
  {func}`fluidimage.works.BaseWorkFromImage.process_1_image`.

- The work class {class}`fluidimage.image2image.Work` has a new method
  {func}`fluidimage.works.image2image.WorkImage2Image.display`.

- The work class {class}`fluidimage.preproc.Work` has a new method
  {func}`fluidimage.works.preproc.WorkPreproc.display`.

## [0.2.0] (2024-02-19)

- Python >=3.9,<3.12
- Better support for Windows and MacOS
- Fix bugs related to subpix and `nb_peaks_to_search`
- Dev and build: PDM, Nox and Meson

## [0.1.5] (2023-02-15)

- Requires Python 3.9
- Improves legend, warnings, error log and documentation

## [0.1.4] (2022-12-13)

- Support Python 3.10
- Avoid a bug with pyfftw 0.13

## [0.1.3] (2021-09-29)

- Many bugfixes!
- Improve VectorFieldOnGrid and ArrayOfVectorFieldsOnGrid
- UVmat compatibility
- Fix incompatibility OpenCV and PyQt5

## [0.1.2] (2019-06-05)

- Bugfix install Windows

## [0.1.1] (2019-05-23)

- Optical flow computation
- Bugfixes + internal code improvements

## 0.1.0 (2018-10-03)

- New topologies and executors with Trio!
- Much better coverage & many bugfixes!
- Better surface tracking

## 0.0.3 (2018-08-29)

- Requirement Python >= 3.6
- Surface tracking
- image2image preprocessing
- BOS topology
- Handle .cine file
- Calibration
- fluidimslideshow-pg and fluidimviewer-pg (based on PyQtgraph)
- OpenCV backend for preprocessing

## 0.0.2 (2017-04-13)

- Bug fixes and documentation changes.
- Continuous integration (python 2.7 and 3.5) with bitbucket pipelines
  ([coverage ~40%](https://codecov.io/gh/fluiddyn/fluidimage))
- Preprocessing of images.
- First simple GUI (`fluidimviewer` and `fluidimlauncher`).

## 0.0.1b (2016-05-31)

- Topology and waiting queues classes to run work in parallel.
- PIV work and topology (multipass, different correlation methods).

[0.1.1]: https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.1.0...0.1.1
[0.1.2]: https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.1.1...0.1.2
[0.1.3]: https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.1.2...0.1.3
[0.1.4]: https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.1.3...0.1.4
[0.1.5]: https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.1.4...0.1.5
[0.2.0]: https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.1.5...0.2.0
[0.3.0]: https://foss.heptapod.net/fluiddyn/fluidimage/-/compare/0.2.0...0.3.0
