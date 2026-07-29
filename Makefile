# cairo lives outside the venv; every render target needs this.
export DYLD_FALLBACK_LIBRARY_PATH := /opt/homebrew/lib

FIGCHECK := uv run figcheck

help:
	@echo 'make test                      pytest'
	@echo 'make check F=figures/x.py      render + all deterministic gates'
	@echo '  F may carry flags: F="figures/x.py --report"'
	@echo '  flags: --report --zoom X0,Y0,X1,Y1[:S] --readback-prompt'
	@echo '         --paper --no-autoplace --width PX'
	@echo 'make regress                   corpus golden diff (exit 1 on drift)'
	@echo 'make update [F=figures/x.py]   refresh committed svg+png baselines'

test:
	uv run pytest -q

check:
	@test -n "$(F)" || { echo 'usage: make check F=figures/<name>.py'; exit 2; }
	$(FIGCHECK) $(F)

regress:
	$(FIGCHECK) --regress

update:
	$(FIGCHECK) --update $(F)

# deprecated alias for `check`
fig: check

.PHONY: help test check regress update fig
.DEFAULT_GOAL := help
