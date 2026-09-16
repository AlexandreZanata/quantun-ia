PYTHON ?= python3

.PHONY: check test lean-plan-check lean-suite-check lean-gate legacy-evidence-check

check:
	$(PYTHON) -m json.tool research/lean/hypotheses.json >/dev/null
	$(PYTHON) -m json.tool research/lean/datasets.json >/dev/null
	$(PYTHON) -m json.tool research/lean/baselines.json >/dev/null
	$(PYTHON) -m json.tool research/lean/smoke/smoke32.json >/dev/null
	$(PYTHON) -m json.tool research/lean/smoke/adversarial.json >/dev/null
	$(PYTHON) scripts/validate_lean_research.py
	$(PYTHON) scripts/lean_verify.py validate-suite
	$(PYTHON) -m py_compile scripts/validate_lean_research.py scripts/lean_verify.py
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest --noconftest -q tests/test_lean_research.py tests/test_lean_verify.py tests/test_ci_contract.py

lean-plan-check:
	$(PYTHON) scripts/validate_lean_research.py

lean-suite-check:
	$(PYTHON) scripts/lean_verify.py validate-suite

lean-gate:
	$(PYTHON) scripts/lean_verify.py run-gate --replay

test: check

legacy-evidence-check:
	$(PYTHON) -m json.tool research/evidence_registry.json >/dev/null
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest --noconftest -q tests/test_evidence_registry.py
