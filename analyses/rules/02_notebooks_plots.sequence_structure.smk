from pathlib import Path


# Separated iDeepS and PrismNet plots ; sequence-only are averaged.
SEPARATE_NAME_NB = "seq_and_structure.separate_prismnet_ideeps.ipynb"
INPUT_SEQSTRUCT_SEPARATE_NB_FILE = Path("notebooks") / SEPARATE_NAME_NB
OUTPUT_SEQSTRUCT_SEPARATE_DIR = OUTPUT_DIR / "sequence_structure"
OUTPUT_SEQSTRUCT_SEPARATE_NB_FILE = OUTPUT_PYSSTER_LENGTH_DIR / SEPARATE_NAME_NB 


# Separated iDeepS and PrismNet plots ; sequence-only are averaged.
#AVG_NAME_NB = "seq_and_structure.avg_prismnet_ideeps.ipynb"
#INPUT_AVG_NB_FILE = Path("notebooks") / NAME_NB
#OUTPUT_AVG_DIR = OUTPUT_DIR / "sequence_structure"
#OUTPUT_AVG_NB_FILE = OUTPUT_PYSSTER_LENGTH_DIR / NAME_NB 


rule run_separate_seqstruct_nb:
    input:
        notebook = INPUT_SEQSTRUCT_SEPARATE_NB_FILE,
        full_summarized_table = FULL_TABLE_FILTERED_FILE,
        config_viz = "config/visualization.yaml"
    output:
        notebook = OUTPUT_SEQSTRUCT_SEPARATE_NB_FILE,
    params:
        notebook_dir = INPUT_PYSSTER_LENGTH_NB_FILE.parent,
        output_dir = OUTPUT_SEQSTRUCT_SEPARATE_DIR,
    shell:
        """papermill {input.notebook} {output.notebook} \
            --cwd {params.notebook_dir} \
            -p export_plots False \
            -p path_output_dir {params.output_dir} \
            ; 
        """