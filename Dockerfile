#base images
FROM ubuntu:18.04
FROM python:3.8

#copy env file
WORKDIR /src

#install dependencies    
RUN pip install aif360 && \
    pip install 'aif360[LFR,OptimPreproc]' && \
    pip install 'aif360[all]'

#clone repo
RUN git clone https://github.com/Trusted-AI/AIF360.git

#copy dataset
COPY aif360/data/raw/meps/h181.csv /src/AIF360/aif360/data/raw/meps/h181.csv
COPY aif360/data/raw/meps/h192.csv /src/AIF360/aif360/data/raw/meps/h192.csv


#change work directory
WORKDIR /src/AIF360

#compile notebooks to run in jupyter lab
RUN pip install --editable '.[all]' && \
    pip install -e '.[notebooks]' && \
    pip install jupyterlab

#exec inside the container and run jupyterlab

