FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base r-cran-randomforest python3.8 python3-pip python3-setuptools python3-dev git


#install dependencies
RUN pip install aif360 && \
    pip install 'aif360[LFR,OptimPreproc]' && \
    pip install 'aif360[all]'

#clone repo
RUN git clone https://github.com/Trusted-AI/AIF360.git

#change work directory
WORKDIR /src/AIF360
RUN Rscript -e "install.packages('data.table')"
