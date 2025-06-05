# Leiden Cell Observatory - OMERO Jupyter Notebooks

This GitHub repository contains Jupyter notebooks to interact with OMERO data in different ways.

## Features
- Prepare AI training data sets using the micro-SAM annotator tool on OMERO datasets and screens. Saving the annotations back to OMERO allows for easy access and reuse of the training data.
- Use the training data to fine-tune micro-SAM models.
- Run inference on OMERO datasets using micro-SAM models.
- Upload metadata from [MIHCSME](https://fairdomhub.org/investigations/575) (Minimal Information for High Content Screening in Microscopy Experiments) Excel files to OMERO, allowing for structured metadata storage and retrieval.

These notebooks are still in an early stage of development and may not be fully functional or stable. Any feedback is highly appreciated.  

# Setting up
These notebooks can either be run locally using conda environments or can be used by installing the docker image provided which provides everything that is needed to run the notebooks on a Jupyter server.

## Local Installation

It is recommended to run these notebooks in a conda environment to make handling the package dependencies easier.

You will need to install conda, if you haven't done yet. It is recommended to use MiniForge for this: [MiniForge](https://github.com/conda-forge/miniforge).

Handling multiple AI tools in a single environment can be difficult, 
hence separate conda environment files are available to should work with the tool and OMERO.

Use on of the environment yaml files to setup the conda environment.
```sh
conda env create -f environment_omero_micro_sam.yml
```

Then activate the environment:

```sh
conda activate micro-sam
```

Then start the Jupyter server:

```sh
jupyter notebook
```

## Docker Installation

We provide a Docker container with all necessary dependencies pre-installed. This is the easiest way to get started.

### Prerequisites

1. Install [Docker](https://www.docker.com/products/docker-desktop/)

2. For GUI applications (cellpose, napari, etc.):
   - **Windows**: Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [MobaXterm](https://mobaxterm.mobatek.net/)
   - **macOS**: Install [XQuartz](https://www.xquartz.org/)
   - **Linux**: X11 is already installed

### Running the Container

Clone this repository and navigate to its directory:

```bash
git clone https://github.com/yourusername/omero-ipynb.git
cd omero-ipynb
```

#### Build the Docker image:

```powershell
docker build -t omero-ipynb .
```

#### Run the container:

For Windows (after starting VcXsrv with "Disable access control" checked):
```powershell
docker run -p 8888:8888 -v ${PWD}/data:/home/jovyan/data -v ${PWD}/notebooks:/home/jovyan/notebooks -v ${PWD}/microsam:/home/jovyan/microsam -e DISPLAY=host.docker.internal:0 --gpus=all  omero-ipynb 
```
Open the Jupyter URL that appears in the console output.

## Micro-SAM
These notebooks are based on [micro-sam](https://github.com/computational-cell-analytics/micro-sam).
The notebooks allow to:
- Annotate datasets stored in OMERO. These annotations are stored with the original data using OMERO.tables, OMERO.ROIs . This structured storing of the annotations allow to train or fine-tune different AI models.
- Finetune micro-sam models
- Run inference on OMERO-data.

For more detailed information, please refer to the [micro-sam README](../microsam/README.md).

## Metadata


## Stardist
Notebooks to apply Stardist segmentation on OMERO datasets (2D,3D and timeseries data).

## Acknowledgments
These notebooks depend on a large number of open-source software packages. Some of the most important:
- [micro-SAM](https://github.com/computational-cell-analytics/micro-sam)
- [Cellpose](https://github.com/MouseLand/cellpose/)
- napari-omero[https://github.com/tlambert03/napari-omero]

## Contact
For questions, reach out to Maarten Paul (m.w.paul@lacdr.leidenuniv.nl). For issues or suggestions, please use the Issues section of the GitHub repository.

This repository is developed with the NL-BioImaging intrastructure, funded by NWO (National Roadmap for Large-Scale Research Facilities).
