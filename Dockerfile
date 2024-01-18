FROM python:3.11.7

RUN pip install --upgrade pip

# install torch
RUN pip install torch torchvision
RUN pip install lightning fastapi uvicorn


