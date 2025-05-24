# Use PyTorch Jupyter notebook with CUDA support
FROM quay.io/jupyter/pytorch-notebook:cuda12-conda-25.3.1

USER root

# Install system dependencies required for micro_sam, image processing, and X11 client libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgeos-dev \
        libgl1-mesa-dev \
        build-essential \
        # X11 client libraries needed for GUI applications
        libx11-6 \
        libxcb1 \
        libxau6 \
        libxdmcp6 \
        libxfixes3 \
        libxcb-shm0 \
        libxcb-render0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable to cache micro_sam models
ENV MICROSAM_CACHEDIR=/home/jovyan/microsam_cache

# Create the cache directory and set permissions
RUN mkdir -p ${MICROSAM_CACHEDIR} && chown -R jovyan:users ${MICROSAM_CACHEDIR}

USER jovyan

# Set conda to not timeout as quickly to help with slow connections
ENV CONDA_FETCH_TIMEOUT=600

# Create micro_sam environment
RUN conda create -n micro-sam python=3.12 -c conda-forge -y && \
    conda install -n micro-sam micro_sam && \
    conda run -n micro-sam pip install https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp312-cp312-manylinux_2_28_x86_64.whl && \
    conda run -n micro-sam pip install ezomero python-dotenv omero-py opencv-python

# Create cellpose environment with GUI support
RUN conda create -n cellpose-env python=3.9 -c conda-forge -y && \
    conda run -n cellpose-env pip install cellpose[gui]

# Create OMERO environment with Python 3.12
RUN conda create -n omero-env python=3.12 -c conda-forge -y && \
    conda run -n omero-env pip install https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp312-cp312-manylinux_2_28_x86_64.whl && \
    conda run -n omero-env pip install ezomero python-dotenv omero-py

# Set environment variable for cellpose models directory
ENV CELLPOSE_LOCAL_MODELS_PATH=/home/jovyan/cellpose_models
RUN mkdir -p ${CELLPOSE_LOCAL_MODELS_PATH}

# Install nb_conda_kernels in the base environment to make conda envs visible to Jupyter
RUN conda install -y -c conda-forge nb_conda_kernels

# Initialize conda for the shell
RUN conda init bash

# Clean up conda cache to reduce image size
RUN conda clean -a -y

# Jupyter is already configured to run in the base image, so no need to explicitly expose ports or set CMD
# The base image already handles these settings:
# EXPOSE 8888
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

WORKDIR /home/jovyan