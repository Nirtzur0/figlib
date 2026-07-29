export DYLD_FALLBACK_LIBRARY_PATH := /opt/homebrew/lib

test:
	uv run pytest -q

fig:
	uv run python -m figlib.cli $(F)

.PHONY: test fig
