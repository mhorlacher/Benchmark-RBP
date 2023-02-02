#!/bin/bash

source $HOME/.bashrc

echo "Creating conda env..."
conda create -n multirbp-cpu python==3.7
echo "Env created. Activating"
conda activate multirbp-cpu

conda config --add channels r
conda config --add channels bioconda

echo "Installing conda packages..."
mamba install pysam
mamba install -c conda-forge pandas numpy

echo "Installing pip packages..."
pip install --upgrade pip
pip install tensorflow==2.0 keras==2.3.1
pip install protobuf==3.20 # downgrade protobuf to avoid error loading keras package
pip install h5py==2.10.0 # downgrade to avoid error loading model during eval

echo "Well done!"