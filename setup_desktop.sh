#!/bin/bash

# Create a directory for desktop shortcuts
mkdir -p /home/ubuntu/Desktop

# Define and create the .desktop files
printf '[Desktop Entry]\nVersion=1.0\nName=Run Napari (Micro-SAM)\nExec=pixi run napari-microsam\nIcon=napari\nType=Application\nCategories=Science;Viewer;Development;\n' > /home/ubuntu/Desktop/napari-microsam.desktop
printf '[Desktop Entry]\nVersion=1.0\nName=Run Cellpose GUI\nExec=pixi run cellpose-gui\nIcon=cellpose\nType=Application\nCategories=Science;Viewer;Development;\n' > /home/ubuntu/Desktop/cellpose-gui.desktop
printf '[Desktop Entry]\nVersion=1.0\nName=Launch Jupyter Lab\nExec=pixi run jupyter-lab\nIcon=jupyter\nType=Application\nCategories=Science;Viewer;Development;\n' > /home/ubuntu/Desktop/jupyter-lab.desktop

# Make the shortcuts executable
chmod +x /home/ubuntu/Desktop/*.desktop