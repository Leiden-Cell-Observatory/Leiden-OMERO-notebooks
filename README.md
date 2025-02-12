# OMERO Jupyter Notebooks

This github repository contains Jupyter notebooks to interact with OMERO data.

The notebooks are still in an early stage, but can be used as inspiration how to analyze data from OMERO.

They show how data can be loaded from OMERO and how results can be stored back into OMERO as attachments or ROIs.

## Installation 
It is recommended to run these notebooks in a conda environment, to make handling the package dependencies easier.

You will need to install conda, recommended to use MiniForge for this:
https://github.com/conda-forge/miniforge

Handling multiple AI tools in one environment can be difficult, hence separate conda environment files are available to should work with the tool and OMERO.

## Metadata

## Stardist

## Micro-sam
These notebooks are based on micro-sam (https://github.com/computational-cell-analytics/micro-sam)

To be able to use the notebooks you need to create a conda environment.
```
conda env create -f environment_omero_micro_sam.yml
```
Then activate the environment.
```
conda activate micro-sam
```
Then start jupyter server
```
jupyter notebook
```

For micro-sam the following routine can be followed.

- Semi manual anotation of data. This can be data on OMERO. The notebook allows to select a random set of images from your dataset. Using a OMERO.table, what images you have annotated can be store
- Finetuning of AI model on your data
- Applying fine-tuned model to your image data.

For questions reach out to Maarten Paul (m.w.paul@lacdr.leidenuniv.nl)
For issues or suggestions please use the Issues section of the GitHub repository.
