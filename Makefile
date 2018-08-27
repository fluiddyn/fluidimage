
.PHONY: clean clean_all develop build_ext_inplace

develop: clean
	python setup.py develop

build_ext_inplace:
	python setup.py build_ext --inplace

clean:
	rm -rf build

clean_so:
	find fluidimage -name "*.so" -delete

cleanall: clean clean_so

black:
	black -l 82 fluidimage

tests:
	OMP_NUM_THREADS=1 python -m unittest discover -v

_tests_coverage:
	mkdir -p .coverage
	OMP_NUM_THREADS=1 coverage run -p -m unittest discover

_report_coverage:
	coverage combine
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

coverage: _tests_coverage _report_coverage
