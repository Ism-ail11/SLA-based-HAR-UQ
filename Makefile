.PHONY: test smoke lint
PYTHON ?= python

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m rateless_har demo --output outputs/demo

lint:
	$(PYTHON) -m ruff check .
