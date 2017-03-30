
PATHPyQt4 = $(shell python -c "import PyQt4; print(PyQt4.__file__.split('PyQt4')[0] + 'PyQt4')")

UI_FILES = $(wildcard *.ui)
PY_FILES = $(UI_FILES:.ui=.py)

.PHONY: run clean

ui2py: $(PY_FILES)

run: $(PY_FILES)
	python -m fluidimage.gui.launcher.main

clean:
	rm -f $(PY_FILES)

%.py: %.ui
	python $(PATHPyQt4)/uic/pyuic.py $(notdir $<) -o $@

