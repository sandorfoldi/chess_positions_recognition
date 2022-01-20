# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

COPY requirements.txt requirements.txt
COPY src/ src/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install gsutil

RUN mkdir data
RUN gsutil -m cp -r gs://chess_predictor_data_small/processed data/

ENTRYPOINT ["python", "src/models/train_model.py"]