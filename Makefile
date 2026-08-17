# Regenerating the results page. Everything else lives in pyproject.toml.
#
#   make demo                 full benchmark, ~6 minutes
#   make demo STRIDE=20       a fast sample, for checking the page renders
#
# Needs three paths, as arguments or environment variables:
#   CHECKPOINT  the USR 2.0 Large .pth
#   CORPUS      directory holding the corpus (see docs/setup.md)
#   CLIP        a video with a real face, for the Occlusion section

CHECKPOINT ?= $(VOXLENS_CHECKPOINT)
CORPUS     ?= $(VOXLENS_CORPUS)
CLIP       ?= $(VOXLENS_TEST_CLIP)
STRIDE     ?= 1
PY         ?= .venv/bin/python

.PHONY: demo check-inputs

check-inputs:
	@test -n "$(CHECKPOINT)" || { echo "set CHECKPOINT (or VOXLENS_CHECKPOINT)"; exit 2; }
	@test -n "$(CORPUS)"     || { echo "set CORPUS (or VOXLENS_CORPUS)"; exit 2; }
	@test -n "$(CLIP)"       || { echo "set CLIP (or VOXLENS_TEST_CLIP)"; exit 2; }

demo: check-inputs
	$(PY) -m voxlens.evaluate "$(CORPUS)" --corpus lrs3 --checkpoint "$(CHECKPOINT)" \
	  --stride $(STRIDE) --out .demo-results.json
	$(PY) -m voxlens.cli "$(CLIP)" --checkpoint "$(CHECKPOINT)" --json > .demo-occlusion.json
	$(PY) scripts/build_demo.py --results .demo-results.json --occlusion .demo-occlusion.json
