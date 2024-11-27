# TidySpot

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
1. Create and activate the Conda environment (may have issues with GraspnetAPI, install it without dependencies):
   ```bash
   conda env create -n tidyspot-env -f ./tidyspot-conda-environment.yml
   conda activate tidyspot-env
   ```
2. Clone and install third party directories, i.e. MinkowskiEngine, GroundedSam, AnyGrasp:
   ```
   git submodule update --init --recursive
   ```
   Follow the instructions in or run install_env.sh
   ```
   ./install_env.sh
   ```
   ### manual install (not preferred)
   ```bash
   cd ~
   git clone https://github.com/NVIDIA/MinkowskiEngine.git
   cd MinkowskiEngine
   export MAX_JOBS=2;
   python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
   ```
3. Clone the AnyGrasp repository and follow install instructions
4. Get license and download checkpoints

### Running the code
Example usage of the TidySpot project with SAM as perception module and antipodal grasp detection
```
python main.py --grasp_type antipodal --perception_type sam --scenario objects/simple_cracker_box_detection_test.yaml --device cpu
```
### Folder structure
    .
    ├── anygrasp_sdk
    │   ├── grasp_detection
    │   │   ├── license                         # License folder
    │   │   ├── checkpoints                     # Create this folder
    │   │   │   ├── checkpoint_detection.tar    # Checkpoint from AnyGrasp (link to download from email)
    │   │   ├── gsnet.so                        # Make sure it is the python 3.10 version
    │   │   ├── lib_cxx.so                      # Make sure it is the python 3.10 version
    |   │   └── ...
    │   └── ...
    ├── main.py
    └── ...

## Notes
### Converting to Jupyter Notebooks
```
pip install jupytext
```
Convert the python file
```
jupytext --to notebook main.py
```