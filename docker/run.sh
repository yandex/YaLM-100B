IMAGE_NAME=yalm-cuda11-ds:1.0

docker run \
--mount type=bind,source=/dev/shm,target=/dev/shm \
-v $HOME:$HOME \
--name "yalm-cuda11-ds-${USER}" \
-v ${SSH_AUTH_SOCK}:${SSH_AUTH_SOCK} -e SSH_AUTH_SOCK="${SSH_AUTH_SOCK}" \
-e REAL_USER="${USER}" \
--net host -it --rm --gpus all \
$IMAGE_NAME /bin/bash
