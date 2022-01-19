# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN mkdir root/data/

COPY requirements.txt root/requirements.txt
COPY setup.py root/setup.py
COPY src/ root/src/
COPY .dvc/ root/.dvc/
COPY data.dvc root/data.dvc

RUN pip install -r root/requirements.txt --no-cache-dir
RUN pip install dvc
RUN dvc pull

ENTRYPOINT ["python", "src/models/train_model.py"]