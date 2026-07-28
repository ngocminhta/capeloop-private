SHELL := /bin/sh
PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help compile doctor config-check contract-check test check

help:
	@printf '%s\n' \
		'make compile        Compile the Python source and tests' \
		'make doctor         Check the runtime and repository configuration' \
		'make config-check   Validate every checked-in TOML configuration' \
		'make contract-check Validate the static mixed-effects contract' \
		'make test           Run the complete offline unittest suite' \
		'make check          Run all universal offline repository checks'

compile:
	$(PYTHON) -m compileall -q src tests

doctor:
	PYTHONPATH=src $(PYTHON) -m cape_loop doctor

config-check:
	@set -eu; \
	find configs -path configs/local -prune -o -type f -name '*.toml' -print | \
	LC_ALL=C sort | \
	while IFS= read -r config; do \
		PYTHONPATH=src $(PYTHON) -m cape_loop config validate "$$config" >/dev/null; \
	done
	@printf '%s\n' 'All checked-in TOML configurations are valid.'

contract-check:
	$(PYTHON) analysis/confirmatory-mixed-effects/validate_contract.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

check: compile doctor config-check contract-check test
