# OMERO micro-sam notebooks

These notebooks are based on [micro-sam](https://github.com/computational-cell-analytics/micro-sam).
WIth micro-sam the following routines are available.

- Semi manual anotation of data. This can be data on OMERO. The notebook allows to select a random set of images from your dataset. Using a OMERO.table, what images you have annotated can be store
- Finetuning of AI model on your data (on OMERO)
- Applying fine-tuned model to your image data (on OMERO)

## Running on image analysis computer (Windows)

These instructions will help to run jupyter notebooks with OMERO at Leiden University

To access the Jupyter notebooks you can either use VS code:

- Open **VS Code** from Start menu, File -> Open Folder
    - make sure Python and Jypyter extensions are installed, VS code will probably ask if neccesary)
    - Open the Notebook you want to use, make sure the right conda environment is selected at the right-top of the notebook.

 - or use **Powershell** to start a Jupyter notebook server:

```bash
cd "E:\[Github repo location]\omero_ipynb"
conda activate micro-sam
jupyter notebook
```

## Setup connection with OMERO
Currently we make use of .env file to store credentials, this is not optimal on a shared computer. Looking for a better solution.

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
