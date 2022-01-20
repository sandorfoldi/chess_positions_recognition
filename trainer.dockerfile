# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

COPY requirements.txt requirements.txt
COPY .dvc .dvc
COPY setup.py setup.py
COPY src/ src/
COPY data.dvc data.dvc

RUN dir

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc
RUN dvc pull

ENTRYPOINT ["python", "app/src/models/train_model.py"]