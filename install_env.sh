python -m venv venv
source venv/bin/activate
pip3 install manipulation --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/
pip install wheel
cd third_party/Grounded-Segment-Anything/
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.8
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade "diffusers[torch]"
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# not too sure if this is needed
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh

# anygrasp
pip install graspnetAPI --no-deps