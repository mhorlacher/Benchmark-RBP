#!/bin/bash


# Increases bin size from 50 to 150 to have the right length for DeepRiPe and Multi-resBind


FILE=$1


CHROMSIZES=$2

bedtools slop -i $FILE -g $CHROMSIZES -b 50