import sys
from pathlib import Path

FULL_TABLE_FILTERED_FILE = "datasets/2023-01-01_benchmark_processed_data/2023-01-27_FullTableAurocFiltered.tsv" 
OUTPUT_DIR = Path("results/2023-01-01_benchmark_results/")

# Contains exploration of the full table, before filtering aberant results.
#include: "rules/02_notebooks_plots.main.smk" 

# Dedicated to Pysster 101 vs Pysster
include: "02_notebooks_plots.pysster_length.smk"

# Dedicated to the comparison of seq+struct and seq-only methods.
include: "02_notebooks_plots.sequence_structure.smk"

# Dedicated to the cross-celltype predictions analysis
#include: "rules/02_notebooks_plots.cross_celltypes.smk"

#
##
rule all:
    input:
        OUTPUT_PYSSTER_LENGTH_NB_FILE,
        OUTPUT_SEQSTRUCT_SEPARATE_NB_FILE,
        #OUTPUT_SEQSTRUCT_AVG_NB_FILE,
        #OUTPUT_CROSS_CT_NB_FILE,
