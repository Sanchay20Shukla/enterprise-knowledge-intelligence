FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 EKI_DATA_DIR=/app/data/runtime
WORKDIR /app
COPY pyproject.toml ./
COPY eki ./eki
RUN pip install --no-cache-dir '.[neural,observability]' && useradd --uid 10001 --create-home app
COPY data/demo ./data/demo
RUN mkdir -p /app/data/runtime && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "eki.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
