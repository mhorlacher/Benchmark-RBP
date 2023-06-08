import sys
from pathlib import Path

##
## 00_PREPROCESSING
## ================
## Dedicated to processing the results tables generated during the benchmark of each method
## into a single table.
## One of the main outputs will be the table of aggregated values (average auROC per method, dataset, fold, etc.)
##


COLUMNS_EVAL_TABLE = [
    'model', 'dataset','RBP_dataset','fold',
    'model_negativeset','sample','prediction','true_class'
]

MAIN_ARCHS = [
    'BERT-RBP',
    'DeepCLIP',
    'DeepRAM',
    'GraphProt',
    'iDeepS', 
    'PRISMNet', # CNN, sequence + structure
    'Pysster', 
    'Pysster-101', # Modified version to test for length impact.
    'RNAProt', # The extended model now!
    'RNAProt-seqonly', # The run from january ; used for comparison against RNAProt
]

# Multilabel methods are quite RAM intensive => separated for easy comment-out.
MAIN_ARCHS += [
    'DeepRiPe', 
    #'DeepRiPe-1_V1',  #TODO: what is this? => Lambert's rerun attempt, useless.
    'DeepRiPe-1', # Complemented with negative-1 class.
    'MultiRBP', 
    #'MultiRBP-1', #TODO: where for January? => MultiRBP was actually very long to train (no early stopping) + perform poorer than e.g. DeepRiPe => not supplemented.
    'Multi-resBind', 
    'Multi-resBind-1',
]

CROSSCT_PROT_ARCHS = [
    "DeepCLIP",
    "iDeepS",
    "Pysster",
    #"Pysster-101",
    "PRISMNet",
]


# January: first submission ; May: review runs including models' mods.
BATCHES = [
    '2023-01-01',
    '2023-05-24',
] 
EXPERIMENTS = [
    'main',
    'crossCT',
    'crossProt',
]

RESULTS_DIR = "results/{BATCH}/00_processed/{EXPERIMENT}/"
INPUT_PER_ARCH_FILE = "datasets/{BATCH}_benchmark_results/{EXPERIMENT}/results.{ARCH}.csv.gz"
OUTPUT_PER_ARCH_FILE = "results/{BATCH}/00_processed/{EXPERIMENT}/aurocs_per_arch/{ARCH}.tsv.gz"
OUTPUT_FILE = "results/{BATCH}/00_processed/{EXPERIMENT}/aurocs.tsv.gz"


def GET_AVAILABLE_ARCHS(**kwargs):
    """Reduce the query list {MAIN_ARCHS} to the set with available input files.
    """
    return [ARCH for ARCH in MAIN_ARCHS
            if Path(INPUT_PER_ARCH_FILE.format(ARCH=ARCH, **kwargs)).exists()
            ]

rule all:
    input:
        # Explicitely separating the geeneration of each set of tables for easy commenting.
        #january_main_aurocs = expand(
        #                        OUTPUT_PER_ARCH_FILE,
        #                        BATCH=['2023-01-01',],
        #                        EXPERIMENT=['main',],
        #                        ARCH=GET_AVAILABLE_ARCHS(BATCH='2023-01-01', EXPERIMENT='main')
        #                        ),
        #january_main_aurocs_gathered = OUTPUT_FILE.format(BATCH='2023-01-01', EXPERIMENT='main'),
        #january_crossCT_aurocs = expand(
        #                            OUTPUT_PER_ARCH_FILE,
        #                            BATCH=['2023-01-01',],
        #                            EXPERIMENT=['crossCT',],
        #                            ARCH=GET_AVAILABLE_ARCHS(BATCH='2023-01-01', EXPERIMENT='crossCT')
        #                            ),
        #january_crossCT_aurocs_gathered = OUTPUT_FILE.format(BATCH='2023-01-01', EXPERIMENT='crossCT'),
        #january_crossProt_aurocs = expand(
        #                            OUTPUT_PER_ARCH_FILE,
        #                            BATCH=['2023-01-01',],
        #                            EXPERIMENT=['crossProt',],
        #                            ARCH=GET_AVAILABLE_ARCHS(BATCH='2023-01-01', EXPERIMENT='crossProt')
        #                            ),
        #january_crossProt_aurocs_gathered = OUTPUT_FILE.format(BATCH='2023-01-01', EXPERIMENT='crossProt'),
        ##
        may_main_aurocs = expand(
                                OUTPUT_PER_ARCH_FILE,
                                BATCH=['2023-05-24',],
                                EXPERIMENT=['main',],
                                ARCH=GET_AVAILABLE_ARCHS(BATCH='2023-05-24', EXPERIMENT='main')
                                ),
        may_main_aurocs_gathered = OUTPUT_FILE.format(BATCH='2023-05-24', EXPERIMENT='main'),
        may_crossCT_aurocs = expand(
                                    OUTPUT_PER_ARCH_FILE,
                                    BATCH=['2023-05-24',],
                                    EXPERIMENT=['crossCT',],
                                    ARCH=GET_AVAILABLE_ARCHS(BATCH='2023-05-24', EXPERIMENT='crossCT')
                                    ),
        may_crossCT_aurocs_gathered = OUTPUT_FILE.format(BATCH='2023-05-24', EXPERIMENT='crossCT'),
        may_crossProt_aurocs = expand(
                                    OUTPUT_PER_ARCH_FILE,
                                    BATCH=['2023-05-24',],
                                    EXPERIMENT=['crossProt',],
                                    ARCH=GET_AVAILABLE_ARCHS(BATCH='2023-05-24', EXPERIMENT='crossProt')
                                    ),
        may_crossProt_aurocs_gathered = OUTPUT_FILE.format(BATCH='2023-05-24', EXPERIMENT='crossProt'),

## rule aggregate_per_arch: calculate summary statistics from each validation set.
## Aggregation results in the main auROC value, in addition to diagnosis statistics
## such as average prediction score per class. 
rule aggregate_per_arch:
    input:
        input_fp = INPUT_PER_ARCH_FILE,
    output:
        output_fp = OUTPUT_PER_ARCH_FILE
    params:
        columns = ",".join(COLUMNS_EVAL_TABLE),
        arch = lambda wildcards: wildcards.ARCH,
        #path_results_table = lambda wildcards: INPUT_PER_ARCH_FILE.format(BATCH=wildcards.BATCH, EXPERIMENT=wildcards.EXPERIMENT, ARCH="{ARCH}"), # partial format trick.
    shell:
        """
        python code/script_aggregate_eval_results_per_arch.py \
            --output_fp {output.output_fp} \
            --path_results_table {input.input_fp} \
            --columns {params.columns} \
            --arch {params.arch}
        """

## rule gather_aggregate_per_arch: generate a single table with all archs.
rule gather_aggregate_per_arch:
    input:
        files = lambda wildcards: expand(
                    OUTPUT_PER_ARCH_FILE,
                        BATCH=wildcards.BATCH,
                        EXPERIMENT=wildcards.EXPERIMENT,
                        ARCH=GET_AVAILABLE_ARCHS(**wildcards))
    output:
        output_fp = OUTPUT_FILE,
    params:
        archs = lambda wildcards: GET_AVAILABLE_ARCHS(**wildcards),
        path_results_table = lambda wildcards: OUTPUT_PER_ARCH_FILE.format(BATCH=wildcards.BATCH, EXPERIMENT=wildcards.EXPERIMENT, ARCH="{ARCH}"), # partial format trick.
    run:
        import pandas as pd  
        import polars as pl 
        import gzip

        import sys
        sys.path.insert(0, "code/")
        import local_code
        from pathlib import Path

        summarized_table_all = []
        for arch in params.archs:
            try:
                summarized_table = pd.read_csv(
                                        str(params.path_results_table).format(ARCH=arch),
                                        sep="\t",
                                        compression="gzip"
                                        )
            except pd.errors.EmptyDataError as e:
                print(f"Empty file for {arch}.")
                raise e
                #continue

            summarized_table_all.append(summarized_table)
        

        # Add unique IDs.
        full_summarized_table = pd.concat(summarized_table_all, axis=0, ignore_index=True)
        full_summarized_table = full_summarized_table.reset_index(drop=True).reset_index().rename(columns={'index':'unique_id'})
        
        # Export.
        full_summarized_table.to_csv(output.output_fp, header=True, index=False, sep="\t", compression="gzip")




# =======
# JANUARY
# =======

#JANUARY_INPUT_MAIN_RESULTS_TABLE_FILE = "datasets/2023-01-01_benchmark_results/main/results.{ARCH}.csv.gz"
#
#JANUARY_MAIN_AVAIL_ARCHS = [ARCH for ARCH in MAIN_ARCHS
#                                if Path(JANUARY_INPUT_MAIN_RESULTS_TABLE_FILE.format(ARCH=ARCH)).exists()
#                                ]
#
#
#JANUARY_RESULTS_DIR = Path("results/2023-01-01/00_processed/")
#
#JANUARY_MAIN_AGGREGATED_FILE = JANUARY_RESULTS_DIR / "main_auroc.tsv.gz"
#JANUARY_MAIN_AGG_PER_ARCH_FILE = JANUARY_RESULTS_DIR / "main_per_arch/{ARCH}.tsv.gz"
#
#include: "00_preprocessing.january.main.smk"
#
#JANUARY_INPUT_CROSSCT_RESULTS_TABLE_FILE = "datasets/2023-01-01_benchmark_results/crossCT/results.{ARCH}.csv.gz"
#
#JANUARY_CROSSCT_AVAIL_ARCHS = [ARCH for ARCH in MAIN_ARCHS
#                                if Path(JANUARY_INPUT_CROSSCT_RESULTS_TABLE_FILE.format(ARCH=ARCH)).exists()
#                                ]
#
#JANUARY_CROSSCT_AGG_PER_ARCH_FILE = JANUARY_RESULTS_DIR / "crossCT_per_arch/{ARCH}.tsv.gz"
#JANUARY_CROSSCT_AGGREGATED_FILE = JANUARY_RESULTS_DIR / "crossCT_auroc.tsv.gz"
#
#include: "00_preprocessing.january.cross_celltypes.smk"
#
#
#
##include: "00_preprocessing.january.cross_protocol.smk"
#
#
## === #
## MAY #
## === #
#
#
#
#MAY_INPUT_MAIN_RESULTS_TABLE_FILE = "datasets/2023-05-24_benchmark_results/main/results.{ARCH}.csv.gz"
#MAY_MAIN_AVAIL_ARCHS = [ARCH for ARCH in ARCHS
#                    if Path(MAY_INPUT_MAIN_RESULTS_TABLE_FILE.format(ARCH=ARCH)).exists()
#                    ]
#
#
#MAY_RESULTS_DIR = Path("results/2023-05-24/")
#MAY_MAIN_AGGREGATED_FILE = MAY_RESULTS_DIR / "main_fullTableAuroc.tsv.gz"
#MAY_MAIN_AGG_PER_ARCH_FILE = MAY_RESULTS_DIR / "main_per_arch/{ARCH}.tsv.gz"
#
#include: "00_preprocessing.may.main.smk"
#
#
#rule all:
#    input:
#        january_main_aggregated = JANUARY_MAIN_AGGREGATED_FILE,
#        may_main_aggregated = MAY_MAIN_AGGREGATED_FILE,
#        #january_cross_protocol_aggregated = JANUARY_CROSS_PROTOCOL_AGGREGATED_FILE,
#        #january_cross_celltype_aggregated = JANUARY_CROSS_CELLTYPE_AGGREGATED_FILE,
#
#
#
