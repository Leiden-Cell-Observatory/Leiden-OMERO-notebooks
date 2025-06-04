# OMERO micro-sam notebooks

These notebooks are based on [micro-sam](https://github.com/computational-cell-analytics/micro-sam).
WIth micro-sam the following routines are available.

## Overview
- Semi manual annotation of data. This can be data on OMERO. The notebook allows to select a random set of images or patches from your dataset. Using a OMERO.table, what images you have annotated will be stored.
- Finetuning of AI model on your data (stored on OMERO).
- Applying fine-tuned model to your image data (on OMERO).

## Prerequisites
- CUDA-capable GPU (recommended)
- OMERO server access (for OMERO-connected workflows)
- Python
- Conda

## Available Notebooks

- environment_omero_micro_sam.yml - Conda environment specification with all required dependencies.
- local-microsam-prediction.ipynb - Run Micro-SAM predictions on local image data.
- local-microsam-training.ipynb - Train Micro-SAM models using local datasets.
- omero-microsam-annotate-batch.ipynb - Create training data by annotating images from OMERO.
- omero-microsam-prediction.ipynb - Apply trained models to images stored in OMERO.
- omero-microsam-training.ipynb - Train Micro-SAM models using OMERO-hosted datasets.

## Running on image analysis computer (Windows)

These instructions will help to run Jupyter notebooks with OMERO at Leiden University

To access the Jupyter notebooks you can either use VS code:

- Open **VS Code** from Start menu, 
    - make sure Python and Jypyter extensions are installed, VS code will probably ask if neccesary)
    - Open the the folder with the notebooks ```File -> Open Folder``` . Select the notebook you want to use, make sure the right conda environment is selected at the right-top of the notebook.

 - or use **Powershell** to start a Jupyter notebook server:

```bash
cd "E:\[Github repo location]\omero_ipynb"
conda activate micro-sam
jupyter notebook
```

## Setup connection with OMERO
Currently we make use of .env file to store credentials, this is not optimal on a shared computer. Looking for a better solution.
For now take the ```.env.example``` file and rename to ```.env``` . Save your login details, remove when you are done.

> **Warning:**
> Ensure that you remove your login and password from the `.env` file when you are done!

## Pull latest version of notebooks from Github

- Open Github Desktop
- Select omero_ipynb repository
- Click Sync

## Installing OMERO - micro-sam
```bash
conda env create -f microsam/environment_omero_micro_sam.yml
```

## Updating micro-sam enviroment:
```bash
conda env update -n micro-sam -f microsam/environment_omero_micro_sam.yml
```
