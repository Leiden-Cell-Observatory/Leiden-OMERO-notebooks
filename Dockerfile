FROM condaforge/miniforge3:latest

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgeos-dev \
    vim \
    gcc \
    g++ \
    zlib1g-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libtiff-dev \
    libwebp-dev \
    libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install BIAFLOWS requirements in base environment
RUN conda install -y python=3.7
RUN pip install git+https://github.com/cytomine-uliege/Cytomine-python-client.git@v2.7.3
RUN pip install git+https://github.com/Neubias-WG5/biaflows-utilities.git@v0.9.2

# Create separate conda environment for micro-sam
RUN conda create -n sam -y 
RUN conda run -n sam conda install -c conda-forge micro_sam

# Copy run script
COPY run.py /app/run.py
COPY descriptor.json /app/descriptor.json

#include the micro-sam models

# Keep container running with base environment (BIAFLOWS)
CMD ["bash"]