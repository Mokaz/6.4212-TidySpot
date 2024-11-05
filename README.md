# TidySpot: Autonomous Room Cleaning with Boston Dynamics’ Spot

## Overview

Robotics has demonstrated immense potential in performing household tasks. 
Advances in mobile manipulation have enabled robots to handle a wide variety of tasks in home environments, and may in time simplify everyday living. This project aims to explore whether Spot from Boston Dynamics can effectively identify and clear clutter in a room with an arbitrary layout. 
This repository aims to successfully simulate Spot using its arm and onboard sensors to detect, grasp, and transport objects from randomly placed positions in a room into a large bin, using the open-soruce Drake Simulator. 
In the Drake simulation, the objects will be of unknown types, and the room may contain multiple obstacles.

## Installation Guide

### Prerequisites
1. **Install CUDA 11.8**  
   Follow the [official CUDA installation guide](https://developer.nvidia.com/cuda-toolkit).
   
2. **Install Conda**  
   [Download Conda](https://docs.anaconda.com/anaconda/install/linux/) and follow the installation instructions for your OS.

### Environment Setup
1. Create and activate the Conda environment:
   ```bash
   conda env create -n tidyspot-env -f ./env-cu118-tidyspot.yml
   conda activate tidyspot-env
2. Clone and install the MinkowskiEngine:
   ```bash
   cd ~
   git clone https://github.com/NVIDIA/MinkowskiEngine.git
   cd MinkowskiEngine
   python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
3. Clone the AnyGrasp repository and follow install instructions
4. Get license and download checkpoints
