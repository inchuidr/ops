FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOWOPS_HOME=/data
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY mlflowops ./mlflowops
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY data ./data
RUN useradd -m app && mkdir /data && chown app /data
USER app
EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "mlflowops.api:app", "--host", "0.0.0.0", "--port", "8000"]
