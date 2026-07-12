.PHONY: dev test test-watch lint lint-fix typecheck coverage dashboard dashboard-local experiment experiment-large repro export-results hpo figures latex-tables release release-check paper-sync paper-build paper-build-publication arxiv-bundle replay-publication replay-publication-artifacts repro-publication-ci open-science-preflight power-analysis microqml-bench publish-leaderboard publish-leaderboard-check check check-real health health-gpu docker-build docker-test docker-lint clean install train-demo nano-parity-bench nano-parity-download nano-parity-publication api api-demo e2e reviewer-repro citation-ready citation-ready-full finalize-citation dvc-check dvc-setup dvc-push model-card exp-026 exp-026-publication exp-024-publication exp-025-publication exp-027 exp-027-publication exp-028 exp-028-publication exp-029 exp-029-publication exp-030 exp-030-publication exp-031 exp-031-publication continuous-train batch-predict data-open-higgs data-open-synthea-cv data-open-nihr-cv data-open-gobug data-open-verify exp-032 exp-032-publication exp-033 exp-033-publication exp-034 exp-034-publication exp-035 exp-035-publication exp-036 exp-036-publication exp-037 exp-037-publication exp-038 exp-038-publication exp-039 exp-039-publication exp-040 exp-040-publication exp-041 exp-041-publication exp-042 exp-042-publication exp-043 exp-043-publication exp-044 exp-044-publication exp-045 exp-045-publication exp-046 exp-046-publication exp-051 exp-051-publication exp-054 exp-054-publication phase-c-publication phase-d-preflight phase-v1.1.0-preflight phase-v1.2.0-preflight ship download-model ship-all-p0 ship-hybrid-higgs

PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
RUFF ?= $(shell test -x .venv/bin/ruff && echo .venv/bin/ruff || echo ruff)

install:
	$(PYTHON) -m pip install -e .

health:
	$(PYTHON) scripts/health_check.py $(HEALTH_OPTS)

health-gpu:
	$(PYTHON) scripts/health_check.py --gpu

check: lint-local typecheck
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/ --ignore=tests/real -m "not real" --cov=src --cov-fail-under=80 -q
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/integration/ tests/contracts/ -m "not real" -q
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/e2e/ -q

check-real:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/real/ -m real $(if $(VERBOSE),,-q)

e2e:
	MLFLOW_DISABLE=1 $(PYTHON) -m pytest tests/e2e/ -v

typecheck:
	$(PYTHON) -m mypy src/training src/quantum

dev:
	docker compose up app

test:
	docker compose run --rm test

test-watch:
	docker compose run --rm app pytest tests/ -v --tb=short -x

lint:
	docker compose -f docker-compose.test.yml run --rm lint

lint-local:
	$(RUFF) check src/ tests/ scripts/

lint-fix:
	docker compose run --rm app ruff check --fix src/ tests/ experiments/

coverage:
	docker compose -f docker-compose.test.yml run --rm test pytest tests/ --ignore=tests/real -m "not real" --cov=src --cov-report=html --cov-report=term-missing -q

dashboard:
	docker compose up dashboard

dashboard-local:
	@$(PYTHON) dashboard/terminal_report.py
	@$(PYTHON) -m streamlit run dashboard/app.py --server.headless true --server.port 8501

model-lab:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m streamlit run dashboard/app.py --server.headless true --server.port 8502

demo-open-predict:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/demo_open_predict.py --rows 5000

demo-open-predict-hybrid:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/demo_open_predict.py --exp-id exp_037 --model-name large_nano_hybrid --rows 5000

demo-cv-clinic:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/demo_cv_clinic.py

demo-cv-game:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/demo_cv_clinic.py --game

export-model-results:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/export_model_results.py --rows 5000

experiment:
	docker compose run --rm experiment

experiment-large:
	QML_PROFILE=publication_large $(PYTHON) scripts/run_publication_large.py

repro:
	MLFLOW_DISABLE=1 $(PYTHON) -m pytest tests/integration/test_exp_001_smoke.py -v

export-results:
	$(PYTHON) scripts/export_results.py

hpo:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/run_hpo.py --exp exp_011_uci_tabular_qml --trials 5 --model perceptron

nas:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_016_hybrid_nas/run.py

poison-topology:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_017_poison_topology/run.py

fusion:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_018_feature_fusion/run.py

train-demo:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.nano_train --model perceptron --dataset breast_cancer --profile ci

nano-parity-bench:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.nano_parity_bench --suite --profile ci

nano-parity-download:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.nano_parity_bench --download-only

nano-parity-publication:
	MLFLOW_DISABLE=1 QML_PROFILE=publication $(PYTHON) experiments/exp_022_nano_quantum_parity/run.py --profile publication --write-results

exp-026:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_026_real_app_e2e/run.py --profile ci

exp-026-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_026_real_app_e2e/run.py --profile publication

exp-024-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_024_quantum_nano_bc/run.py --profile publication --write-results --write-model-card

exp-025-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_025_pima_generalization/run.py --profile publication --write-results

exp-027:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_027_continuous_retrain/run.py --profile ci

exp-027-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_027_continuous_retrain/run.py --profile publication --write-results

continuous-train:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/continuous_train.py --profile publication

exp-028:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_028_chatbot_tool_parity/run.py --profile ci

exp-028-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_028_chatbot_tool_parity/run.py --profile publication --write-results

exp-029:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_029_batch_calc_parity/run.py --profile ci

exp-029-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_029_batch_calc_parity/run.py --profile publication --write-results

exp-030:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_030_publication_large/run.py --profile ci

exp-030-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_030_publication_large/run.py --profile publication_large --write-results

exp-031:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_031_curriculum_clinical/run.py --profile ci

exp-031-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_031_curriculum_clinical/run.py --profile publication --write-results

exp-032:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_032_large_nano_higgs/run.py --profile ci

exp-032-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_032_large_nano_higgs/run.py --profile publication --write-results

exp-060:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_060_large_nano_acyd_soy/run.py --profile ci

exp-060-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_060_large_nano_acyd_soy/run.py --profile publication --write-results

exp-081:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_081_large_nano_acyd_maize/run.py --profile ci

exp-081-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_081_large_nano_acyd_maize/run.py --profile publication --write-results

exp-062:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_062_hybrid_nano_acyd_soy/run.py --profile ci

exp-062-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_062_hybrid_nano_acyd_soy/run.py --profile publication --write-results

exp-063:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_063_quantum_warmstart_acyd/run.py --profile ci
exp-063-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_063_quantum_warmstart_acyd/run.py --profile publication --write-results

exp-064:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_064_entangle_schedule_acyd/run.py --profile ci
exp-064-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_064_entangle_schedule_acyd/run.py --profile publication --write-results

exp-065:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_065_gv_alr_hybrid_acyd/run.py --profile ci
exp-065-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_065_gv_alr_hybrid_acyd/run.py --profile publication --write-results

exp-066:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_066_noise_reg_acyd/run.py --profile ci
exp-066-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_066_noise_reg_acyd/run.py --profile publication --write-results

exp-067:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_067_reupload_ladder_acyd/run.py --profile ci
exp-067-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_067_reupload_ladder_acyd/run.py --profile publication --write-results

exp-068:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_068_nano_grand_comparison/run.py --profile ci --write-results

exp-068-publication:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_068_nano_grand_comparison/run.py --profile publication --write-results

exp-068a:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_068a_angle_encoding_acyd/run.py --profile ci

exp-068a-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_068a_angle_encoding_acyd/run.py --profile publication --write-results

exp-068b:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_068b_compound_stress_acyd/run.py --profile ci

exp-068b-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_068b_compound_stress_acyd/run.py --profile publication --write-results

exp-061:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_061_conventional_acyd_baselines/run.py --profile ci

exp-061-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_061_conventional_acyd_baselines/run.py --profile publication --write-results

exp-076:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_076_conventional_nihr_baselines/run.py --profile ci

exp-076-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_076_conventional_nihr_baselines/run.py --profile publication --write-results

exp-077:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_077_conventional_gobug_baselines/run.py --profile ci

exp-077-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_077_conventional_gobug_baselines/run.py --profile publication --write-results

exp-069:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_069_large_nano_nihr/run.py --profile ci

exp-069-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_069_large_nano_nihr/run.py --profile publication --write-results

exp-070:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_070_large_nano_gobug/run.py --profile ci

exp-070-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_070_large_nano_gobug/run.py --profile publication --write-results

exp-071:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_071_hybrid_nano_gobug/run.py --profile ci

exp-071-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_071_hybrid_nano_gobug/run.py --profile publication --write-results

exp-033:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_033_higgs_serve_parity/run.py --profile ci

exp-033-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_033_higgs_serve_parity/run.py --profile publication --write-results

exp-034:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_034_large_nano_synthea/run.py --profile ci

exp-034-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_034_large_nano_synthea/run.py --profile publication --write-results

exp-035:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_035_synthea_serve_parity/run.py --profile ci

exp-035-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_035_synthea_serve_parity/run.py --profile publication --write-results

exp-036:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_036_method_ablation_higgs/run.py --profile ci

exp-036-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_036_method_ablation_higgs/run.py --profile publication --write-results

exp-037:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_037_hybrid_nano_higgs/run.py --profile ci

exp-037-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_037_hybrid_nano_higgs/run.py --profile publication --write-results

exp-038:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_038_hybrid_serve_parity/run.py --profile ci

exp-038-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_038_hybrid_serve_parity/run.py --profile publication --write-results

exp-039:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_039_synthea_regularized/run.py --profile ci

exp-039-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_039_synthea_regularized/run.py --profile publication --write-results

exp-040:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_040_full_scale_ablation_higgs/run.py

exp-040-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_040_full_scale_ablation_higgs/run.py --write-results

exp-041:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_041_human_cv_clinical_cases/run.py

exp-041-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_041_human_cv_clinical_cases/run.py --write-results

exp-042:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_042_sample_scale_precision/run.py

exp-042-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_042_sample_scale_precision/run.py --write-results

exp-043:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_043_calibration_synthea/run.py

exp-043-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_043_calibration_synthea/run.py --write-results

exp-044:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_044_nihr_cv_baseline/run.py

exp-044-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_044_nihr_cv_baseline/run.py --profile publication --write-results

exp-045:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_045_code_defect_gobug/run.py

exp-045-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_045_code_defect_gobug/run.py --profile publication --write-results

exp-051:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_051_quantum_head_nihr/run.py

exp-051-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_051_quantum_head_nihr/run.py --profile publication --write-results

exp-054:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_054_adaptive_hybrid_higgs/run.py

exp-054-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_054_adaptive_hybrid_higgs/run.py --profile publication --write-results

exp-052:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_052_quantum_warmstart_higgs/run.py

exp-052-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_052_quantum_warmstart_higgs/run.py --profile publication --write-results

exp-072:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_072_quantum_warmstart_nihr/run.py --profile ci

exp-072-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_072_quantum_warmstart_nihr/run.py --profile publication --write-results

exp-073:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_073_quantum_warmstart_gobug/run.py --profile ci

exp-073-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_073_quantum_warmstart_gobug/run.py --profile publication --write-results

exp-074:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_074_entangle_schedule_nihr/run.py --profile ci

exp-074-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_074_entangle_schedule_nihr/run.py --profile publication --write-results

exp-075:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_075_adaptive_hybrid_nihr/run.py --profile ci

exp-075-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_075_adaptive_hybrid_nihr/run.py --profile publication --write-results

exp-078:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_078_agro_clinical_cases/run.py

exp-078-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_078_agro_clinical_cases/run.py --write-results

exp-080:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_080_quantum_champion_fusion_acyd/run.py --profile ci

exp-080-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_080_quantum_champion_fusion_acyd/run.py --profile publication --write-results

exp-079:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_079_quantum_transfer_higgs_to_acyd/run.py --profile ci

exp-079-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_079_quantum_transfer_higgs_to_acyd/run.py --profile publication --write-results

exp-082:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_082_calibration_acyd/run.py --profile ci

exp-082-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_082_calibration_acyd/run.py --profile publication --write-results

exp-053:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_053_entangle_schedule_bc/run.py

exp-053-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_053_entangle_schedule_bc/run.py --profile publication --write-results

exp-055:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_055_noise_reg_gobug/run.py

exp-055-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_055_noise_reg_gobug/run.py --profile publication --write-results

exp-056:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_056_reupload_curriculum_ladder/run.py

exp-056-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_056_reupload_curriculum_ladder/run.py --profile publication --write-results

exp-057:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_057_param_shift_ablation/run.py

exp-057-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_057_param_shift_ablation/run.py --profile publication --write-results

exp-046:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_046_model_scale_curve/run.py

exp-046-publication:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) experiments/exp_046_model_scale_curve/run.py --profile publication --write-results

batch-predict:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/batch_predict.py \
		--input tests/fixtures/breast_cancer_holdout.csv \
		--output .local/out/probabilities.csv \
		--exp-id quantum_nano_bc_app

data-open-higgs:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/build_open_higgs.py

data-open-synthea-cv:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/build_synthea_cv_risk.py

data-open-nihr-cv:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/build_nihr_cv_synthetic.py

data-open-gobug:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/build_gobug_subset.py

data-open-acyd-download:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/download_acyd_brazil.py --crop soybean

data-open-acyd-soy: data-open-acyd-download
	MLFLOW_DISABLE=1 $(PYTHON) scripts/build_open_acyd_soy.py

data-open-acyd-maize-download:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/download_acyd_brazil.py --crop maize

data-open-acyd-maize: data-open-acyd-maize-download
	MLFLOW_DISABLE=1 $(PYTHON) scripts/build_open_acyd_maize.py

data-open-acyd-dvc:
	@if $(PYTHON) -m dvc --version >/dev/null 2>&1; then \
		$(PYTHON) -m dvc add data/open/acyd_soy_brazil/processed/v1; \
	else \
		echo "ERROR: DVC not installed. Run: $(PYTHON) -m pip install dvc && make data-open-acyd-dvc"; \
		exit 1; \
	fi

data-open-verify:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/validate_open_data.py $(if $(DATASET),--dataset $(DATASET),)
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/contracts/test_open_data_manifest.py -q

phase-c-publication: exp-024-publication exp-025-publication
	@echo "Phase C publication refresh complete (RTX 4060)"

phase-d-preflight: check-real citation-ready-full open-science-preflight publish-leaderboard publish-leaderboard-check
	@echo "Phase D local preflight complete — pending manual Zenodo DOI + arXiv ID"
	@echo "Next: git tag v1.0.0 && git push origin v1.0.0"
	@echo "Then: make finalize-citation DOI=10.5281/zenodo.XXXXXXX [ARXIV_ID=2606.XXXXX]"

phase-v1.1.0-preflight: health-gpu check-real
	@echo "v1.1.0 preflight complete (RTX 4060) — Phases F+G+H+I+J, 12/12 real gate"
	@echo "Next: git tag v1.1.0 && git push origin v1.1.0"

phase-v1.2.0-preflight: health-gpu data-open-verify exp-033-publication check-real
	@echo "v1.2.0 preflight complete (RTX 4060) — Phase L5 serve parity, 14/14 real gate"
	@echo "Next: git tag v1.2.0 && git push origin v1.2.0"

train-ship:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/demo_predict.py --profile publication --epochs 30 --rows 5

demo-predict:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) scripts/demo_predict.py --profile mini --epochs 12 --rows 3

api:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.api_server --host 127.0.0.1 --port 8000

api-demo:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_020_api_smoke/run.py

figures:
	$(PYTHON) scripts/generate_figures.py

latex-tables:
	$(PYTHON) scripts/export_latex_tables.py

release:
	$(PYTHON) scripts/prepare_release.py

release-check:
	$(PYTHON) scripts/prepare_release.py --verify-only dist/release

open-science-preflight:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/open_science_preflight.py

paper-sync:
	$(PYTHON) scripts/build_paper.py --sync-only

paper-artifacts: latex-tables figures paper-sync
	@echo "Paper artifacts ready (tables + figures synced to paper/)"

paper-build: paper-artifacts
	$(PYTHON) scripts/build_paper.py

paper-build-publication:
	@mkdir -p logs
	cp tests/contracts/fixtures/publication_experiments.jsonl logs/experiments.jsonl
	$(MAKE) paper-artifacts

phase-6-publication: citation-ready exp-068-publication paper-artifacts paper-build
	@echo "Phase 6 complete — paper/main.pdf + citation-ready (skip-release)"

repro-publication-ci:
	MLFLOW_DISABLE=1 bash scripts/repro_publication_ci.sh

arxiv-bundle: paper-build
	$(PYTHON) scripts/prepare_arxiv_submission.py

arxiv-bundle-sources:
	$(PYTHON) scripts/prepare_arxiv_submission.py --skip-pdf-check

reviewer-repro:
	MLFLOW_DISABLE=1 bash scripts/reviewer_repro.sh

citation-ready:
	$(PYTHON) scripts/validate_citation_ready.py --skip-release

citation-ready-full: release release-check
	$(PYTHON) scripts/validate_citation_ready.py

finalize-citation:
	@test -n "$(DOI)" || (echo "Usage: make finalize-citation DOI=10.5281/zenodo.XXXXXXX [ARXIV_ID=2606.XXXXX]" && exit 1)
	$(PYTHON) scripts/finalize_citation.py --doi "$(DOI)" $(if $(ARXIV_ID),--arxiv-id "$(ARXIV_ID)",)

dvc-check:
	$(PYTHON) scripts/validate_dvc.py

dvc-setup:
	$(PYTHON) -m pip install -q dvc
	$(PYTHON) scripts/dvc_remote_setup.py

dvc-push:
	$(PYTHON) scripts/dvc_push.py --setup-remote

dvc-push-full:
	$(PYTHON) scripts/dvc_push.py --setup-remote --replay

replay-publication:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/replay_publication.py

replay-publication-artifacts:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/replay_publication.py --artifacts-only

power-analysis:
	$(PYTHON) scripts/power_analysis.py --table

microqml-bench:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/export_microqml_bench.py

publish-leaderboard:
	$(PYTHON) scripts/publish_leaderboard.py

publish-leaderboard-check:
	$(PYTHON) scripts/publish_leaderboard.py --verify-only

export-reference-datasets:
	$(PYTHON) scripts/export_reference_datasets.py

model-card:
	$(PYTHON) scripts/generate_model_card.py

ship:
	@test -n "$(MODEL)" || (echo "Usage: make ship MODEL=large_nano_mlp_synthea [PROFILE=publication] [RETRAIN=1] [SKIP_TRAIN=1] [SKIP_GATE=1]" && exit 1)
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model $(MODEL) \
		$(if $(PROFILE),--profile $(PROFILE),) \
		$(if $(RETRAIN),--retrain,) \
		$(if $(SKIP_TRAIN),--skip-train,) \
		$(if $(SKIP_GATE),--skip-gate,)

download-model:
	@test -n "$(MODEL)" || (echo "Usage: make download-model MODEL=large_nano_mlp_synthea" && exit 1)
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.qml_download --model $(MODEL)

ship-all-p0:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_synthea --profile ci --skip-train --skip-gate
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_higgs --profile ci --skip-train --skip-gate
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_acyd_soy --profile ci --skip-train --skip-gate
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_nihr --profile ci --skip-train --skip-gate
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_gobug --profile ci --skip-train --skip-gate
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model quantum_nano_bc --profile ci --skip-train --skip-gate
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_synthea_calibrated --profile ci --skip-train --skip-gate

ship-acyd-soy:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_acyd_soy --profile publication --skip-train

ship-nihr:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_nihr --profile ci --skip-train --skip-gate

ship-gobug:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_mlp_gobug --profile ci --skip-train --skip-gate

ship-hybrid-higgs:
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m scripts.qml_ship --model large_nano_hybrid_higgs --profile publication --skip-train

experiments-new:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/run_exp_011_015.py --profile publication

results-new:
	$(PYTHON) scripts/generate_results_md.py

docker-build:
	docker compose build

docker-test:
	docker compose -f docker-compose.test.yml run --rm test

docker-lint:
	docker compose -f docker-compose.test.yml run --rm lint

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
