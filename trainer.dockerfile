# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN mkdir app

COPY requirements.txt app/requirements.txt
COPY .dvc app/.dvc
COPY setup.py app/setup.py
COPY src/ app/src/
COPY data.dvc app/data.dvc

RUN dir



RUN pip install -r app/requirements.txt --no-cache-dir
# RUN pip install dvc
# RUN dvc pull

ENTRYPOINT ["python", "app/src/models/train_model.py"]