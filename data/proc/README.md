## Benchmarking ENCODE Data Preprocessing

A pipeline to preprocess ENCODE peaks, split them into CV set, generate positive and negative training samples, and finally generate input for different methods. 

### Sub-Workflows

This directory contains a couple of sub-workflows, responsible for different parts of the data preprocessing. 

#### 1. `peaks-cv/peaks/`: Generating high-confidence Peaks
This workflow preprocesses raw ENCODE peaks int high-confidence crosslink sites for each RBP_CELL. 
The result is a set of crosslink site BED files (`peaks/processed/{RBP_CELL}/peaks.crosslink.bed`). 

#### 2. `peaks-cv/`: Splitting Crosslink Sites into CV Folds
This workflow split the crosslink sites BED files into *N* CV folds (`peaks-cv/processed/{RBP_CELL}/fold-{FOLD_DIR}/peaks.crosslink.fold-{FOLD}.bed`). 

#### 3. `train-smaples/`: Generating Positive and Negative Samples
This workflow generates the positive and negative sample set(s) --> `train-samples/processed/{RBP_CELL}/fold-{FOLD_DIR}/{TYPE}.fold-{FOLD}.bed`

#### 4. `train-inputs/`: Generate Final Inputs for each Method
This workflow takes the single-nucleotide positive and negative samples and generates proper inputs for each method. 
Output files are of the form `train-inputs/processed/{RBP_CELL}/{METHOD}/fold-{FOLD}/{TYPE}.fold-{FOLD}.*`. 

---

In general, the workflows should be run in chronological order to ensure all dependencies exist. 