# OMERO Jupyter Notebooks

This GitHub repository contains Jupyter notebooks to interact with OMERO data.

The notebooks are still in an early stage but can be used as inspiration on how to analyze data from OMERO. They show how data can be loaded from OMERO and how results can be stored back into OMERO as attachments or ROIs.

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
docker run -p 8888:8888 -v ${PWD}/data:/home/jovyan/data -v ${PWD}/notebooks:/home/jovyan/notebooks -v ${PWD}/microsam:/home/jovyan/microsam -e DISPLAY=host.docker.internal:0 omero-ipynb 
```

Open the Jupyter URL that appears in the console output.

## Manual Installation

It is recommended to run these notebooks in a conda environment to make handling the package dependencies easier.

You will need to install conda. It is recommended to use MiniForge for this: [MiniForge](https://github.com/conda-forge/miniforge).

Handling multiple AI tools in a single environment can be difficult, 
hence separate conda environment files are available to should work with the tool and OMERO.

Check respective folders for details.

## Micro-sam
These notebooks are based on [micro-sam](https://github.com/computational-cell-analytics/micro-sam).

For more detailed information, please refer to the [micro-sam README](../microsam/README.md).

To use the notebooks, you need to create a conda environment:

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
## Metadata


## Stardist


```

For questions, reach out to Maarten Paul (m.w.paul@lacdr.leidenuniv.nl). For issues or suggestions, please use the Issues section of the GitHub repository.
