
.PHONY: clean cleanall develop

develop: sync

sync:
	pdm sync --no-self
	pdm sync --clean -v

lock:
	pdm lock

clean:
	rm -rf build

cleantransonic:
	pdm run transonic-clean-dir src

cleanall: clean cleantransonic

black:
	pdm run black

isort:
	pdm run isort

format:
	pdm run format

test:
	OMP_NUM_THREADS=1 pdm run pytest src

cov:
	# much slower with TRANSONIC_NO_REPLACE but more accurate
	TRANSONIC_NO_REPLACE=1 OMP_NUM_THREADS=1 pytest --pyargs fluidimage --cov --no-cov-on-fail

cov-html:
	coverage html
