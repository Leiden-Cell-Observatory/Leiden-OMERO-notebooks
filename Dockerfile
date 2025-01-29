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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    # Create working directory
WORKDIR /app

RUN conda create -c conda-forge -n  micro-sam -y micro_sam python=3.11
SHELL ["conda", "run", "-n", "micro-sam", "/bin/bash", "-c"]
RUN conda install pip
RUN conda install -c conda-forge pillow>=10.1


RUN pip install git+https://github.com/cytomine-uliege/Cytomine-python-client.git@v2.7.3
RUN pip install git+https://github.com/Neubias-WG5/biaflows-utilities.git@v0.9.2


# Keep container running for interactive development
CMD ["bash"]


#conda activate micro-sam
#pip install 'Pillow>=10.1'
#pip install git+https://github.com/Neubias-WG5/biaflows-utilities.git@v0.9.2 --no-dependencies
#pip install cytomine-python-client==2.7.3


