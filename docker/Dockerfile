FROM nvcr.io/nvidia/pytorch:20.11-py3

# Though already installed, pybind11 could not be found during the build. Reinstalling fixes the problem.
RUN python3 -mpip install pybind11==2.6.2 -U

##############################################################################
# Updating NCCL to 2.10.3-1
##############################################################################
RUN wget --quiet --tries=5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libnccl2_2.10.3-1+cuda11.0_amd64.deb
RUN apt install --yes --fix-broken --no-install-recommends ./libnccl2_2.10.3-1+cuda11.0_amd64.deb
RUN rm -f ./libnccl2_2.10.3-1+cuda11.0_amd64.deb
RUN wget --quiet --tries=5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libnccl-dev_2.10.3-1+cuda11.0_amd64.deb
RUN apt install --yes --fix-broken --no-install-recommends ./libnccl-dev_2.10.3-1+cuda11.0_amd64.deb
RUN rm -f ./libnccl-dev_2.10.3-1+cuda11.0_amd64.deb

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
        apt-get update && \
        apt-get install -y git && \
        git --version

##############################################################################
# Reinstall PyTorch to 1.10.0 from Oct, 2021; use local NCCL
##############################################################################
RUN git clone --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git checkout v1.10.0 && \
    git submodule sync && \
    git submodule update --init --recursive && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    USE_SYSTEM_NCCL=1 python3 setup.py install && \
    cd .. && rm -rf pytorch

##############################################################################
# apex a651e2c24ecf97cbf367fd3f330df36760e1c597
##############################################################################
RUN git clone --recursive https://github.com/NVIDIA/apex && \
        cd apex && \
        git reset --hard a651e2c24ecf97cbf367fd3f330df36760e1c597 && \
        pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
         cd .. && rm -rf apex

##############################################################################
# Other packages
##############################################################################
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen \
        libaio-dev

##############################################################################
# Installing DeepSpeed
##############################################################################
RUN pip install deepspeed==0.6.5