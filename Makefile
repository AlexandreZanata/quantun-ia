PYTHON ?= .venv/bin/python

.PHONY: check test lean-plan-check lean-gate evidence-monitor exp-011 exp-012 exp-024-ci exp-025-ci

check:
	$(PYTHON) -m json.tool research/evidence_registry.json >/dev/null
	$(PYTHON) -m json.tool research/lean/smoke/smoke32.json >/dev/null
	$(PYTHON) -m json.tool research/lean/smoke/adversarial.json >/dev/null
	$(PYTHON) scripts/validate_lean_research.py
	$(PYTHON) scripts/lean_verify.py validate-suite
	$(PYTHON) -m py_compile dashboard/app.py tests/test_evidence_registry.py scripts/lean_verify.py
	$(PYTHON) -m pytest -q tests/test_evidence_registry.py tests/test_lean_research.py tests/test_lean_verify.py

lean-plan-check:
	$(PYTHON) scripts/validate_lean_research.py

lean-gate:
	$(PYTHON) scripts/lean_verify.py run-gate --replay

test: check

evidence-monitor:
	.venv/bin/streamlit run dashboard/app.py

exp-011:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_011_uci_tabular_qml/run.py

exp-012:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_012_mnist_pca_qml/run.py

exp-024-ci:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_024_quantum_nano_bc/run.py --profile ci

exp-025-ci:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_025_pima_generalization/run.py --profile ci
