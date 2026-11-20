# On Windows without make, run the commands in the README directly.
PY ?= python

install:      ; $(PY) -m pip install -r requirements.txt
test:         ; ROUTER_EMBEDDER=hash $(PY) -m pytest -q
models:       ; $(PY) -m scripts.check_models
data:         ; $(PY) -m scripts.build_benchmark
audit:        ; $(PY) -m scripts.audit
judge-check:  ; $(PY) -m scripts.judge_validation export --n 100
train:        ; $(PY) -m scripts.train_router
eval:         ; $(PY) -m scripts.evaluate_router
onnx:         ; $(PY) -m gateway.onnx_export
serve:        ; uvicorn gateway.app:app --port 8000
loadtest:     ; $(PY) -m loadtest.bench_concurrency --target route gateway direct-cheap direct-strong
shadow:       ; $(PY) -m scripts.shadow_report
dashboard:    ; streamlit run gateway/dashboard.py
# Offline end-to-end demo on SYNTHETIC data (numbers are meaningless; it proves the plumbing works)
demo:
	ROUTER_DB=data/synthetic.db ROUTER_EMBEDDER=hash ROUTER_RESULTS=results/demo ROUTER_ARTIFACT=gateway/models/demo.joblib \
	  sh -c "$(PY) -m scripts.make_synthetic && $(PY) -m scripts.train_router && $(PY) -m scripts.evaluate_router"
docker:       ; docker build -t llm-router .
.PHONY: install test models data audit judge-check train eval onnx serve loadtest shadow dashboard demo docker
