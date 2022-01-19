# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir app/data/

COPY requirements.txt app/requirements.txt
COPY setup.py app/setup.py
COPY src/ app/src/
COPY .dvc/ app/.dvc/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc
RUN dvc pull


ENTRYPOINT ["python", "src/models/train_model.py"]