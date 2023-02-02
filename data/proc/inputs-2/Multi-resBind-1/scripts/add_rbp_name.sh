#!/bin/bash


FILENAME=$1

RBP=$2

awk -v r=$RBP '{print $1"\t"$2"\t"$3"\t"r"\t"$5"\t"$6}' $FILENAME
