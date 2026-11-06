PYTHON ?= .venv/bin/python
PYTEST = $(PYTHON) -m pytest

test-fast:
	$(PYTEST) -m "unit or component"

test-integration:
	$(PYTEST) -m integration

test-all:
	$(PYTEST)

.PHONY: test-fast test-integration test-all
