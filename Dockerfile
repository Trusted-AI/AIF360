

ARG OWNER=jupyter
ARG BASE_CONTAINER=$OWNER/minimal-notebook:python-3.8.8
FROM $BASE_CONTAINER

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN git clone https://github.com/Trusted-AI/AIF360.git
RUN pip install aif360
