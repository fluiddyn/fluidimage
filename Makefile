
.PHONY: clean cleanall develop

develop: sync

sync:
	pdm sync

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
