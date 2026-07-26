SHELL := /bin/sh
PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help doctor test check

help:
	@printf '%s\n' \
		'make doctor  Check the runtime and repository configuration' \
		'make test    Run the complete offline unittest suite' \
		'make check   Run all required local and CI checks'

doctor:
	PYTHONPATH=src $(PYTHON) -m cape_loop doctor

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

check: doctor test
