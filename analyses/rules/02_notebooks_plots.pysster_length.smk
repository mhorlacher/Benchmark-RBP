from pathlib import Path

## PYSSTER LENGTH
## ==============
## Plot loss in performance of Pysster 101b vs Pysster.

INPUT_PYSSTER_LENGTH_NB_FILE = Path("notebooks/pysster_input_length.ipynb")
OUTPUT_PYSSTER_LENGTH_DIR = OUTPUT_DIR / "pysster_length"
OUTPUT_PYSSTER_LENGTH_NB_FILE = OUTPUT_PYSSTER_LENGTH_DIR / "pysster_input_length.ipynb"

#OUTPUT_PYSSTER_LENGTH_PERF_TABLE_FILE = OUTPUT_PYSSTER_LENGTH_DIR / "pysster_prismnet_auroc_table.tsv"
#OUTPUT_PYSSTER_LENGTH_DELTA_FILE = OUTPUT_PYSSTER_LENGTH_DIR / "pysster_prismnet_auroc_table.tsv"

#
## rule extract_table_pysster_length: from the full table of aggregated perf, extract the relevant lines.
## Namely: extract the regular Pysster, Pysster 101, and PrismNet as another reference.
#rule extract_table_pysster_length:
#    input:
#        full_table_filtered = FULL_TABLE_FILTERED_FILE,
#    output:
#        auroc_table = OUTPUT_PYSTER_LENGTH_FILE,
#        delta_auroc_table = OUTPUT_PYSSTER_LENGTH_DELTA_FILE
#    run:
#        import pandas as pd 
#        import polars as pl 
#        import numpy as np 
#
#        full_summarized_table = pd.read_csv(
#                                    input.full_table_filtered,
#                                    sep='\t',
#                                    header=0,
#                                    )
#
#        # This table is used for the 
#        # Export.
#        tmp_pysster_size.to_csv(output.delta_auroc_table, sep='\t', index=False, header=True)
#        tmp_pysster_prismnet.to_csv(output.auroc_table, sep='\t', index=False, header=True)



#
## rule run_pysster_length_nb:
rule run_pysster_length_nb:
    input:
        notebook = INPUT_PYSSTER_LENGTH_NB_FILE,
        full_summarized_table = FULL_TABLE_FILTERED_FILE,
        config_viz = "config/visualization.yaml"
    output:
        notebook = OUTPUT_PYSSTER_LENGTH_NB_FILE,
    params:
        notebook_dir = INPUT_PYSSTER_LENGTH_NB_FILE.parent,
        output_dir = OUTPUT_PYSSTER_LENGTH_DIR,
    shell:
        """papermill {input.notebook} {output.notebook} \
            --cwd {params.notebook_dir} \
            -p export_plots False \
            -p path_output_dir {params.output_dir} \
            ; 
        """

            ##-p path_pysster_prismnet_auroc_file {input.auroc_table} \
            ##-p path_pysster_delta_file {input.delta_auroc_table} ;