## Data

---

### Instruction on How to Run:

1. Obtain the human reference `hg38.fasta`and `hg19.fasta` as well as the corresponding `*.fai` and `*.genomefile` files and move them to `meta/`. 
2. Run the workflow(s) for pre-processing the individual dataset(s) in `datasets/{DATASET}`. 
3. Run the sample input processing workflow in `proc/` (or and of the subworkflows in `proc/peaks-cv/`, `proc/train-samples/` or `proc/train-inputs/`). 
4. You can find the processed inputs for each dataset, method, experiment and CV-fold in `proc/inputs/`

*Note: Check out the `config.yml` files for each workflow for tweaking them, e.g. some workflows allow inclusion of only specific datasets and methods.*