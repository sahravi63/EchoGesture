# Build from the repository root:
#   docker build -f src/Dockerfile -t echogesture .
#   docker run --rm -v %cd%/test_videos:/videos -v %cd%/results:/app/results echogesture --videos /videos
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src/requirements.txt src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt
COPY src/ src/
ENTRYPOINT ["python", "src/main.py"]
