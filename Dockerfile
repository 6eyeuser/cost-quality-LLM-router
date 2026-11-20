# Gateway image (ONNX embedder, no torch).  Build after training:
#   python -m gateway.onnx_export && ROUTER_EMBEDDER=onnx python -m scripts.train_router
#   docker build -t llm-router . && docker run -p 8000:8000 -e GROQ_API_KEY=... llm-router
FROM python:3.12-slim
WORKDIR /app
ENV WORKERS=1 PYTHONUNBUFFERED=1 ROUTER_EMBEDDER=onnx ROUTER_LOG_DB=/data/router_v2.db
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY router ./router
COPY gateway ./gateway
VOLUME /data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
# Routing is CPU-bound Python: scale with processes (WORKERS ~ number of cores), not threads.
CMD uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}
