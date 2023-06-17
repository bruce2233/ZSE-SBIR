# sudo docker run --rm -it --name vigorous_elbakyan --runtime=nvidia --gpus all nvidia/cuda:12.1.0-devel-ubuntu20.04 bash
docker build -t my .

sudo docker run --rm -it --name vigorous_elbakyan --runtime=nvidia --gpus all my bash

docker cp /mnt/g/github/download/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh vigorous_elbakyan:/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

pip install torch==1.10.0+cu113 --index-url  http://172.17.0.1:8080 --trusted-host 172.17.0.1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 

conda create -n zse-sbir python=3.6



