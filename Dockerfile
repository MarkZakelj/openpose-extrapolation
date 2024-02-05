# Dockerfile for dockerized inference
FROM python:3.11.7

RUN pip install --upgrade pip

# install torch
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir lightning fastapi uvicorn

WORKDIR /app/src

COPY src /app/src
COPY models /app/models

EXPOSE 3000

# add --workers <num> to uvicorn command to run with <num> workers
CMD uvicorn predict:app --host 0.0.0.0 --port 3000


