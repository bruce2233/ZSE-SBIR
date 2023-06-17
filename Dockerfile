FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
COPY . /app/ZSE-SBIR/
COPY ./docker-cuda-based/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh /app/
WORKDIR /app
# RUN bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b
RUN bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init && . ~/.bashrc &&  echo $PATH
WORKDIR /app/ZSE-SBIR

RUN . ~/.bashrc && echo $PATH && conda create -n zse-sbir python=3.6 && conda activate zse-sbir && pip install torch==1.10.0+cu113 --index-url  http://172.17.0.1:8080 --trusted-host 172.17.0.1 && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 