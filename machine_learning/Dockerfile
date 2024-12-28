FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04 AS runtime

WORKDIR /workspace

# set home to workdir, so that routines that write to it don't crash
ENV HOME /workspace/

# make sure packages don't ask questions during installation
ARG DEBIAN_FRONTEND=noninteractive

# avoids UTF8 issues
ENV LANG C.UTF-8

# silence warnings about git not being installed
ENV GIT_PYTHON_REFRESH quiet

# required for some reason to make docker build work on our Ubuntu workstation
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt install -y --no-install-recommends python3.8 python3.8-distutils curl

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py; python3.8 get-pip.py

COPY ./ /workspace/

# RUN pip install --no-cache-dir --disable-pip-version-check -r requirements-torch-cu113.txt
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

RUN chmod a+w /workspace/ -R
