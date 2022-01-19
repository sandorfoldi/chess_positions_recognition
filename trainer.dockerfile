# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .dvc/ .dvc/

RUN dir

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc
RUN dvc pull

ENTRYPOINT ["python", "src/models/train_model.py"]