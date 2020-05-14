FROM centos:7

COPY ./scripts/tf_cnn_benchmarks /tf_cnn

WORKDIR /tf_cnn
RUN mkdir train

RUN yum install -y centos-release-scl
RUN yum install rh-python36 -y devtoolset-8

RUN yum update -y && yum install -y \
    ca-certificates \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    which \
    python-setuptools \
    cmake tkinter numactl \
    libSM libXext \
    openmpi-devel \
    openssh openssh-server

RUN source scl_source enable devtoolset-8 rh-python36 && \
    pip install --upgrade pip && \
	pip install tensorflow==1.15.0 requests absl-py
