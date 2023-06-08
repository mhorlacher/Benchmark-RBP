import sys
from pathlib import Path 

## 
## 00_PREPROCESSING.JANUARY_NEGATIVE1
## ==================================
## Dedicated to parsing the January results batch to extract the negative-1 results.
## These models have not changed in the May batch, nor the datasets, hence the results can be reused.
## This is true only for a subset of methods. The other methods have to be re-run for both negative-1 and negative-2
## (e.g. DeepCLIP had its internal ratio train/validation changed for the May batch)
##

COLUMNS_EVAL_TABLE = [
    'model', 'dataset','RBP_dataset','fold',
    'model_negativeset','sample','prediction','true_class',
]

INPUT_PER_ARCH_JANUARY_FILE = "datasets/2023-01-01_benchmark_results/main/results.{ARCH}.csv.gz"
INPUT_PER_ARCH_NEG2_MAY_FILE = "datasets/2023-05-24_benchmark_results/main/negative-2_reruns/results.{ARCH}.csv.gz"
OUTPUT_PER_ARCH_NEG1_JANUARY_FILE = "datasets/2023-05-24_benchmark_results/main/negative-1_january_runs_extracted/results.{ARCH}.csv.gz"
OUTPUT_PER_ARCH_STITCHED_NEG1_NEG2_FILE = "datasets/2023-05-24_benchmark_results/main/complete_stitched_neg1Jan_neg2May/results.{ARCH}.csv.gz"

METHODS = [
    "DeepRAM",
    "GraphProt",
    "iDeepS",
    "RNAProt-seqonly",
]

LIST_METHODS_AVAIL = [method for method in METHODS
                        if (
                            Path(INPUT_PER_ARCH_JANUARY_FILE.format(ARCH=method)).exists()
                            and
                            Path(INPUT_PER_ARCH_NEG2_MAY_FILE.format(ARCH=method)).exists()
                            )]

rule all:
    input:
        extracted = expand(OUTPUT_PER_ARCH_NEG1_JANUARY_FILE, ARCH=LIST_METHODS_AVAIL),
        stitched = expand(OUTPUT_PER_ARCH_STITCHED_NEG1_NEG2_FILE, ARCH=LIST_METHODS_AVAIL)



rule extract_january_negative1:
    input:
        january_source_file = INPUT_PER_ARCH_JANUARY_FILE,
    output:
        january_neg1_extracted = OUTPUT_PER_ARCH_NEG1_JANUARY_FILE,
    params:
        columns = COLUMNS_EVAL_TABLE,
    run:
        import pandas as pd 
        import polars as pl 
        import gzip 

        d = pl.read_csv(input.january_source_file, separator=",", has_header=False).to_pandas()
        d.columns = params.columns

        negative1_extracted = []

        grouping = ["dataset", "RBP_dataset", "fold", "model_negativeset"]

        for group, group_df in d.groupby(grouping):
            if group[-1] == 'negative-1':
                negative1_extracted.append(group_df)

        negative1_extracted_df = pd.concat(negative1_extracted)

        # Export
        negative1_extracted_df.to_csv(output.january_neg1_extracted, header=False, index=False, compression="gzip")


rule stitch_neg1jan_neg2may:
    input:
        january_neg1_extracted = OUTPUT_PER_ARCH_NEG1_JANUARY_FILE,
        may_neg2_rerun = INPUT_PER_ARCH_NEG2_MAY_FILE,
    output:
        stitched = OUTPUT_PER_ARCH_STITCHED_NEG1_NEG2_FILE,
    params:
        columns = COLUMNS_EVAL_TABLE,
    run:
        import pandas as pd 
        import numpy as np 

        neg1 = pd.read_csv(input.january_neg1_extracted, sep=",", header=None, names=params.columns)
        neg2 = pd.read_csv(input.may_neg2_rerun, sep=",", header=None, names=params.columns)

        # Stitch
        stitched = pd.concat([neg1, neg2])
        
        # Export
        stitched.to_csv(output.stitched, header=False, index=False, compression="gzip")
#
#
