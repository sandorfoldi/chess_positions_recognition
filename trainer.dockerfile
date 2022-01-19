# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN mkdir data/

COPY requirements.txt root/requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .dvc/ .dvc/
COPY data.dvc data.dvc

RUN dir

RUN pip install -r root/requirements.txt --no-cache-dir
RUN pip install dvc
RUN dvc pull

ENTRYPOINT ["python", "src/models/train_model.py"]