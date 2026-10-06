FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub==0.30.2

COPY docker/assets-init/init_assets.py /app/init_assets.py

ENTRYPOINT ["python", "/app/init_assets.py"]
