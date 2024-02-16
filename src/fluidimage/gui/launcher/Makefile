
UI_FILES = $(wildcard *.ui)
PY_FILES = $(UI_FILES:.ui=.py)

.PHONY: run clean

run: $(PY_FILES)
	python -m fluidimage.gui.launcher.main

ui2py: $(PY_FILES)

clean:
	rm -f $(PY_FILES)

%.py: %.ui
	pyside6-uic $(notdir $<) -o $@
