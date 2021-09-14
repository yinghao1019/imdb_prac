#BUILD　container for imdb_prac to training model
#on Google Vertex AI service
#Author:Ying Hao Hung F108157110@nkust.wdu.tw

FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl && \
     rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.7 && \
    rm -rf /var/lib/apt/lists/*

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.7 get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

WORKDIR /root

#Intall Model Framework
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
                -f https://download.pytorch.org/whl/torch_stable.html

#copy local project to container
COPY . /root/imdb_prac/

#change to DIR
WORKDIR /root/imdb_prac

#install required dependencies for running app
RUN  pip install -r requirements.txt 

#set variable for load cloud data & run
ENV CLOUD_BUCKET imdbml
ENV CLOUD_ML_PROJECT_ID mlprac-321407
ENV CLOUD_TRAINDATA_PATH gs://${CLOUD_BUCKET}/Data/imdb_train.csv
ENV CLOUD_TESTDATA_PATH gs://${CLOUD_BUCKET}/Data/imdb_test.csv
ENV CLOUD_TOKENIZER_PATH gs://${CLOUD_BUCKET}/tokenizer/imdb_token.txt
ENV PYTHONPATH .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT [ "python3.7","./imdb_prac/task.py"]

#Sets up deafult training arg
CMD ["--warm_ep","5","--batch_size","32","--ep","10"]

