
.PHONY: clean cleanall develop

develop: sync

sync:
	pdm sync

lock:
	pdm lock

clean:
	rm -rf build

cleantransonic:
	find fluidimage -type d -name __pythran__ | xargs rm -rf

cleanall: clean cleantransonic

black:
	pdm run black

isort:
	pdm run isort

test:
	OMP_NUM_THREADS=1 pdm run pytest src

list-sessions:
	@nox --version 2>/dev/null || pip install nox
	@$(NOX) -l
