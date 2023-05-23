import sys
from pathlib import Path

FULL_TABLE_FILTERED_FILE = "datasets/2023-01-01_benchmark_processed_data/2023-01-27_FullTableAurocFiltered.tsv" 
OUTPUT_DIR = Path("results/2023-01-01_benchmark_results/")

#include: "rules/02_notebooks_plots.main.smk"
include: "02_notebooks_plots.pysster_length.smk"
#include: "rules/02_notebooks_plots.cross_celltypes.smk"

#
##
rule all:
    input:
        OUTPUT_PYSSTER_LENGTH_NB_FILE,
        #OUTPUT_CROSS_CT_NB_FILE,
