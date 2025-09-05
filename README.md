# Leiden Cell Observatory - OMERO Jupyter Notebooks

This GitHub repository contains Jupyter notebooks to interact with OMERO data in different ways.

## Features
- Prepare AI training data sets using the micro-SAM annotator tool on OMERO datasets and screens. Saving the annotations back to OMERO allows for easy access and reuse of the training data.
- Use the training data to fine-tune micro-SAM models.
- Run inference on OMERO datasets using micro-SAM models.
- Upload metadata from [MIHCSME](https://fairdomhub.org/investigations/575) (Minimal Information for High Content Screening in Microscopy Experiments) Excel files to OMERO, allowing for structured metadata storage and retrieval.

These notebooks are still in an early stage of development and may not be fully functional or stable. Any feedback is highly appreciated.  

# Setting up
These notebooks can be run either locally by setting up the environments with Pixi, or by using the provided Docker image, which includes a Jupyter server and all necessary dependencies.

## Local Installation (Recommended for development)

We use [Pixi](https://pixi.sh/) to manage the project's dependencies. Pixi simplifies environment management and ensures reproducibility across different platforms (Windows, Linux, macOS).

1.  **Install Pixi:**
    If you haven't already, install Pixi on your machine.
    ```sh
    # On most systems
    curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | bash
    ```
    For more installation options, see the [Pixi Installation Guide](https://pixi.sh/latest/getting-started/installation/).

2.  **Set up the Environments:**
    The project uses a single `pixi.toml` file to define all environments. The `pixi install` command will resolve and download all necessary packages.
    ```sh
    pixi install --all
    ```
    This command will also create a `pixi.lock` file, which guarantees that everyone running the project will have the exact same package versions.

3.  **Run the Notebooks:**
    You can start the Jupyter server from any of the defined environments. For example, to use the environment for micro-sam:
    ```sh
    # To start the Jupyter server with the micro-sam environment
    pixi run --environment microsam jupyter lab
    ```
    To run a different environment, simply change the `--environment` flag to `cellpose`, `full`, or `default`.

## Docker Installation

We provide a Docker container with all necessary dependencies and a pre-configured graphical desktop. This is the easiest way to get started and run GUI-based notebooks like `napari`.

### Prerequisites

1.  Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  To view graphical applications like `napari` from the container, a VNC client is needed.
    - **Windows**: Install a VNC client like [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [MobaXterm](https://mobaxterm.mobatek.net/).
    - **macOS**: Install [XQuartz](https://www.xquartz.org/).
    - **Linux**: X11 is typically pre-installed.

### Building and Running the Container

1.  **Clone the repository and navigate to its directory:**
    ```sh
    git clone [https://github.com/yourusername/omero-ipynb.git](https://github.com/yourusername/omero-ipynb.git)
    cd omero-ipynb
    ```

2.  **Build the Docker image:**
    This command builds the container with all dependencies pre-installed.
    ```sh
    docker build -t omero-notebooks-xfce -f Dockerfile_xfce_gpu .
    ```

3.  **Run the container:**
    This command runs the container with GPU support and exposes the necessary ports.
    ```sh
    docker run -it --rm --gpus all -p 8888:8888 -p 6080:6080 omero-notebooks-xfce
    ```
    - The `--gpus all` flag enables GPU acceleration.
    - The `-p 8888:8888` flag maps the Jupyter Lab port.
    - The `-p 6080:6080` flag maps the VNC server port for the desktop.

4.  **Access the server:**
    - **Jupyter Lab:** Open your web browser and navigate to the URL provided in the console output (e.g., `http://127.0.0.1:8888/...`).
    - **VNC Desktop:** Use your VNC client to connect to `localhost:6080` to access the XFCE desktop.

## Micro-SAM
These notebooks are based on [micro-sam](https://github.com/computational-cell-analytics/micro-sam).
The notebooks allow to:
- Annotate datasets stored in OMERO. These annotations are stored with the original data using OMERO.tables, OMERO.ROIs. This structured storing of the annotations allows to train or fine-tune different AI models.
- Finetune micro-sam models.
- Run inference on OMERO-data.

For more detailed information, please refer to the [micro-sam README](../microsam/README.md).

## Metadata


## Stardist
Notebooks to apply Stardist segmentation on OMERO datasets (2D, 3D, and timeseries data).

## Acknowledgments
These notebooks depend on a large number of open-source software packages. Some of the most important:
- [micro-SAM](https://github.com/computational-cell-analytics/micro-sam)
- [Cellpose](https://github.com/MouseLand/cellpose/)
- [napari-omero](https://github.com/tlambert03/napari-omero)

## Contact
For questions, reach out to Maarten Paul (m.w.paul@lacdr.leidenuniv.nl). For issues or suggestions, please use the Issues section of the GitHub repository.

This repository is developed with the NL-BioImaging infrastructure, funded by NWO (National Roadmap for Large-Scale Research Facilities).