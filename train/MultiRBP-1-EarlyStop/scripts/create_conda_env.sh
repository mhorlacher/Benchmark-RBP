#!/bin/bash

source $HOME/.bashrc

#conda create -n multirbp-gpu-2 python==3.7
conda activate multirbp-gpu-2
conda config --add channels r
conda config --add channels bioconda
mamba install pysam
mamba install -c conda-forge pandas numpy

mamba install -c conda-forge tensorflow-gpu=2.0.0
