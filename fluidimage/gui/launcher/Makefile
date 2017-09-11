
# PYQT=PyQt4
# PATHPyQt4 = $(shell python -c "import $(PYQT); print($(PYQT).__file__.split('$(PYQT)')[0] + '$(PYQT)')")

UI_FILES = $(wildcard *.ui)
PY_FILES = $(UI_FILES:.ui=.py)

.PHONY: run clean

run: $(PY_FILES)
	python -m fluidimage.gui.launcher.main

ui2py: $(PY_FILES)

clean:
	rm -f $(PY_FILES)

%.py: %.ui
	# python $(PATHPyQt4)/uic/pyuic.py $(notdir $<) -o $@
	pyuic5 $(notdir $<) -o $@
