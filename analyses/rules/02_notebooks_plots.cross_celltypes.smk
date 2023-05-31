## CROSS CELLTYPES
## ===============
## Evaluating performance of models trained on ENCODE datasets from one cell-type,
## tested on the other (e.g. Pysster QKI_HepG2 is tested on QKI_K562)

# performance tables:
#TODO: move to filtered tables.
INPUT_MAIN_FILE = "results/{BATCH}/00_processed/main/aurocs.tsv.gz"
INPUT_CROSSCT_FILE = "results/{BATCH}/00_processed/crossCT/aurocs.tsv.gz"

INPUT_CROSS_CT_NB_FILE = Path("notebooks") / ... #TODO:

OUTPUT_CROSS_CT_DIR = OUTPUTDIR / "cross_celltypes"
OUTPUT_CROSS_CT_NB_FILE = OUTPUT_CROSS_CT_DIR / ... #TODO:



## rule run_cross_celltype_nb:
rule run_cross_celltype_nb:
    input:
        notebook = INPUT_CROSS_CT_NB_FILE,
        main_summarized_table = INPUT_MAIN_FILE,
        crossct_summarized_table = INPUT_CRSSCT_FILE,
    output:
        notebook = OUTPUT_CROSS_CT_NB_FILE,
    params:
        config_viz = "config/visualization.yaml",
        outputdir = OUTPUT_CROSS_CT_DIR,
    shell:
        """papermill {input.notebook} {output.notebook} \
            --cwd {params.notebook_dir} \
            -p export_plots False \
            -p path_output_dir {params.output_dir} \
            -p path_main_summarized_table {input.main_summarized_table} \
            -p path_crossct_summarized_table {input.crossct_summarized_table} ;
            ; 
        """