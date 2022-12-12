
.PHONY: clean clean_all develop build_ext_inplace requirements

develop:
	pip install -e .[dev]

build_ext_inplace:
	python setup.py build_ext --inplace

clean:
	rm -rf build

cleanso:
	find fluidimage -name "*.so" -delete

cleantransonic:
	find fluidimage -type d -name __pythran__ | xargs rm -rf

cleanall: clean cleanso cleantransonic

black:
	black -l 82 fluidimage try *.py doc

isort:
	isort -rc --atomic -tc fluidimage bin bench doc/examples

tests:
	OMP_NUM_THREADS=1 pytest

_tests_coverage:
	mkdir -p .coverage
	TRANSONIC_NO_REPLACE=1 OMP_NUM_THREADS=1 coverage run -p -m pytest

_report_coverage:
	coverage combine
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

coverage: _tests_coverage _report_coverage

list-sessions:
	@nox --version 2>/dev/null || pip install nox
	@$(NOX) -l

requirements: 'pip-compile(main)' 'pip-compile(doc)' 'pip-compile(test)' 'pip-compile(dev)'

# Catch-all target: route all unknown targets to nox sessions
%: Makefile
	@nox --version 2>/dev/null || pip install nox
	@nox -s $@
