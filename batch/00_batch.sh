#!/bin/bash
#SBATCH -J jupyter
#SBATCH -t 7-00:00:00
#SBATCH -o /mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/3_project_HCCmove/batch/logs/%A.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user jihu6033@gmail.com
#SBATCH -p TitanRTX
#SBATCH --gpus 1


export HTTP_PROXY=http://192.168.45.108:3128
export HTTPS_PROXY=http://192.168.45.108:3128
export http_proxy=http://192.168.45.108:3128
export https_proxy=http://192.168.45.108:3128


# Define vars
DOCKER_IMAGE_NAME="kimjihoo/hccmovement"
DOCKER_CONTAINER_NAME="kimjihoo_hccmovement2"
RANDOM_PORT=8271
GPU_NAME=" "

docker build -t ${DOCKER_IMAGE_NAME} -f hccmove.Dockerfile .
docker run -it --rm --device=nvidia.com/gpu=all --shm-size 1TB \
    --name "${DOCKER_CONTAINER_NAME}" \
    -e JUPYTER_ENABLE_LAB=yes \
    -p ${RANDOM_PORT}:${RANDOM_PORT} \
    -v /mnt:/workspace \
    -e HTTP_PROXY=${HTTP_PROXY} \
    -e HTTPS_PROXY=${HTTPS_PROXY} \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    ${DOCKER_IMAGE_NAME} \
    bash -c "jupyter notebook --ip=0.0.0.0 --port=${RANDOM_PORT} --no-browser --allow-root --NotebookApp.token=''"
