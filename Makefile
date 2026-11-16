PYTHON ?= python
REPORTS := $(basename $(notdir $(wildcard reports/*.tex)))

.PHONY: help reports check clean

help:
	@echo "make reports   build every report PDF in reports/"
	@echo "make check     byte-compile all scripts"
	@echo "make clean     remove LaTeX build files"

reports: $(addprefix reports/,$(addsuffix .pdf,$(REPORTS)))

reports/%.pdf: reports/%.tex
	cd reports && pdflatex -interaction=nonstopmode -halt-on-error $*.tex >/dev/null
	cd reports && if grep -q '\\bibliography{' $*.tex; then bibtex $* >/dev/null; fi
	cd reports && pdflatex -interaction=nonstopmode -halt-on-error $*.tex >/dev/null
	cd reports && pdflatex -interaction=nonstopmode -halt-on-error $*.tex >/dev/null
	@echo "built $@"

check:
	$(PYTHON) -m py_compile scripts/*.py scripts/neuronpedia/*.py
	@find scripts -name __pycache__ -type d -exec rm -rf {} +
	@echo "all scripts compile"

clean:
	cd reports && rm -f *.aux *.log *.out *.toc *.bbl *.blg
