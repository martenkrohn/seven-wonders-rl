PYTHON ?= .venv/bin/python
PYTEST=$(PYTHON) -m pytest

test-fast:
	$(PYTEST) -m "unit or component" -q

test-integration:
	$(PYTEST) -m integration -q

test-all:
	$(PYTEST) -m "unit or component or integration" -q

.PHONY: test-fast test-integration test-all
