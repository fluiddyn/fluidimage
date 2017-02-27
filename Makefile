
.PHONY: clean clean_all develop build_ext_inplace

develop:
	python setup.py develop

build_ext_inplace:
	python setup.py build_ext --inplace

clean:
	rm -rf build

clean_so:
	find fluidimage -name "*.so" -delete

cleanall: clean clean_so

tests:
	python -m unittest discover

tests_coverage:
	mkdir -p .coverage
	coverage run -m unittest discover
	mpirun -np 2 coverage run -m unittest discover
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"
