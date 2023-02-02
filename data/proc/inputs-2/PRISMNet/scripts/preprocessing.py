"""
Preprocess an example bed file in a format readable by PrismNet
"""

import os
import argparse
import json

import pandas as pd
import numpy as np

import pyBigWig

pd.options.mode.chained_assignment = None  # default='warn'


# Example .bed file for positive sites
# chr1    10019768        10019769        .       1       +
# chr1    10019802        10019803        .       1       +
# chr1    15344736        15344737        .       1       +
# chr1    15394009        15394010        .       1       +

# Example fasta sequence
#  >chr1:7834432-7834532
# GTTGTTTTGTGTGAGTGTTTTGTTGTTGTTGTTGTTGTTTTGACACAAGGTGTCACTCTGTTACCCAGGCTAGAGTGCACTGATGCAATCATAGCTCACT
# >chr1:10019719-10019819
# GTCTTTCTGGTGAGCAGCCCCAACCCTGAGTCACATCATTAGCATAGACTCTGGTGTGTTCTAAAGGGGCTCCTTATGAATAGCAAAAGACAATCCTATC

# Take example data as input (e.g. positive sites)
# peaks = BedTool("../example_data/positive.fold-0.bed")

MAX_LENGTH = 101

def compute_icshape_intervals(bw, d, chr, start, end):
    """Return icshape scores within a given interval i.e. [start, end]
    Args:
        bw (_type_): BigWig file containing icshape scores
        d (_type_): dictionary of chr# -> chr_size
        chr (_type_): chr of the BS
        start (_type_): start of BS
        end (_type_): end of BS
    Returns:
        _type_: _description_
    """

    chr_size = d[chr]

    # try:
    #     if chr_size <= end:
    #         icshape_scores = bw.values(chr, start, end - 1)
    #     else: 
    #         print("Warning: chr_size > end")
    #         icshape_scores = np.append(bw.values(chr, start, end - 1), [np.nan] * (MAX_LENGTH - (end + 1 - start)))

    try:
        icshape_scores = bw.values(chr, start, end)

    except RuntimeError as err:
        print(err)
        print("Additional info: ")
        print("chr: ", chr)
        print("chr size: ", chr_size)
        print("start: ", start)
        print("end: ", end)
        icshape_scores = [np.nan] * MAX_LENGTH

    assert(len(icshape_scores) == MAX_LENGTH)

    return icshape_scores

def add_icshape(bed_df, bw_plus, bw_minus, genomefile):
    """[summary]
    Args:
        bed_df (Pandas DataFrame): [description]
        bw_plus (bigWig): [description]
        bw_minus (bigWig): [description]
    Returns:
        Pandas DataFrame: Returns dataframe enriched in icshape scores.
    """
    # Filter bed by strand
    pos_strand = bed_df[bed_df["strand"] == "+"]
    neg_strand = bed_df[bed_df["strand"] == "-"]

    # Read chromosome sizes from genomefile
    chr_size_df = pd.read_csv(genomefile, sep="\t", header=None)
    chr_size_df = chr_size_df[chr_size_df.iloc[:,0].str.startswith("chr")]
    # Build a dictionary of 'chr#' : 'chr_size'
    keys = chr_size_df.iloc[:,0].to_list()
    values = chr_size_df.iloc[:,1].to_list()
    d = dict(zip(keys,values))

    cols = ["chr", "start_pos", "end_pos"]

    # Add icshape corresponding to pos and negative strand to the bed file
    pos_strand['icshape'] = pos_strand.loc[:,cols].apply(lambda row: compute_icshape_intervals(bw_plus, d, row[0], row[1], row[2]), axis=1).to_numpy()
    neg_strand['icshape'] = neg_strand.loc[:,cols].apply(lambda row: compute_icshape_intervals(bw_minus, d, row[0], row[1], row[2]), axis=1).to_numpy()

    output_df = pd.concat([pos_strand, neg_strand], axis=0)

    print("start_pos:", output_df['start_pos'].iloc[0])
    print("end pos: ", output_df['end_pos'].iloc[0])
    print("len icshape scores: ", len(output_df['icshape'].iloc[0]))

    return output_df

def add_seq(bed_df, fasta):
    """[summary]
    Args:
        bed_df ([type]): [description]
        fasta ([type]): [description]
    Returns:
        [type]: [description]
    """

    # Add sequence info
    sequences = []

    with open(fasta) as f:
        lines = f.readlines()
        print(len(lines))
        i = 1
        for l in lines:
            if i % 2 == 0:
                sequences.append(l.strip())
            i += 1

        assert(len(sequences) * 2 == len(lines))
        assert(len(bed_df) == len(sequences)) 

        bed_df['seq'] = sequences

        return bed_df

def replace_nan(l):
    """Convert list to Numpy Array and replace NaNs with -1
    Args:
        l ([type]): [description]
    Returns:
        [type]: [description]
    """
    arr = np.array(l)
    arr[np.isnan(arr)] = -1
    return arr

def array_to_string(arr):
    string = ",".join([el for el in arr])
    return string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing for PrismNet")

    parser.add_argument('--input_bed', type=str, help="Input .bed file containing positives or negatives")
    parser.add_argument('--input_fasta', type=str, help="Sequences corresponding to positives or negatives")
    parser.add_argument('--icshape_dir', type=str, help="Path to folder containing icSHAPE data")
    parser.add_argument('--genomefile', type=str, help="Path to chromsizes file")
    parser.add_argument('--keep_highest_scoring', action="store_true", help="Whether to filter binding sites to keep only the top (5K) scoring")
    parser.add_argument('--filter_icshape', action="store_true", help="Whether to filter binding sites to have a sufficient icshape coverage")
    parser.add_argument('--ionmf_json', type=str, help="Mapping file (.json dict) from UNKNOWN RBP datasets to known cell lines, for iONMF dataset only (patch)")
    parser.add_argument('--output_tsv', type=str, help="Path to output .tsv file")

    args = parser.parse_args()
    print("############## Arguments ##############")
    print(args)
    print("#######################################")

    splits = args.input_bed.split("/")

    input_dir_list = splits[:-1]
    rbp = input_dir_list[2].split("_")[0]
    cell_line = input_dir_list[2].split("_")[1]
    icshape_cell_line = cell_line

    # Patched, input folders should be renamed
    if cell_line == "UNKNOWN":
        with open(args.ionmf_json) as f:
            ionmf_dict = json.loads(f.read())
            mapped_rbp_dataset = ionmf_dict[input_dir_list[2]]
            print(f"Warning: iONMF folder with unknown cell line {input_dir_list[2]}, remapping to {mapped_rbp_dataset}")
            rbp = mapped_rbp_dataset.split("_")[0]
            icshape_cell_line = mapped_rbp_dataset.split("_")[1]

    fname = splits[-1]

    # Intersect bed peaks and icshape for positive and negative samples
    # Read peaks file
    df = pd.read_csv(args.input_bed, sep='\t', names=["chr", "start_pos", "end_pos", "name", "score", "strand"])
    if len(df) == 0:
        print("Warning: reading empty .bed file, check?")
    print(df.head())

    # Add sequences
    df = add_seq(df, args.input_fasta)
    print(df.head())

    # Read icSHAPE data (bigWig)
    # processed/{DATASET}/{EXPERIMENT}/PrismNet/fold-{FOLD}/  -> EXPERIMENT = RBP_CELL
    icshape_minus = None ; icshape_plus = None
    try:
        icshape_minus = pyBigWig.open(os.path.join(args.icshape_dir, icshape_cell_line + "-minus.bw"))
        icshape_plus = pyBigWig.open(os.path.join(args.icshape_dir, icshape_cell_line + "-plus.bw"))
    except RuntimeError as e:
        print("icSHAPE data not found")
    
    if icshape_plus is None or icshape_minus is None:
    # In case icshape data are not available for this cell line - add missing values
        df['icshape'] = [MAX_LENGTH * [-1] for i in df.index]
    else:
        # Add icshape scores
        df = add_icshape(df, icshape_plus, icshape_minus, args.genomefile)
    
    print(df.head())

    # Add labels as ground truth
    if fname.startswith("positive"):
        df['label'] = 1
        df.replace({'score' : '.'}, 1, inplace=True)
    else:
        df['label'] = 0
        df.replace({'score' : '.'}, -1, inplace=True)

    # Fix data types
    df['score'] = df['score'].astype(float)

    if args.keep_highest_scoring:
        # Keep only top 5000 binding sites with the highest signals
        df = df.sort_values(by=['score'])
        df = df.iloc[:5000]

    if args.filter_icshape:
        # Keep only bs corresponding to at least a 40% icshape coverage
        df['icshape_cov'] = df['icshape'].apply(lambda x: np.count_nonzero(~np.isnan(x))).to_numpy()
        df = df[df['icshape_cov'] >= 40]
    else:
        # Placeholder
        df['icshape_cov'] = -1

    # Replace nan icshape scores with -1
    df['icshape'] = df['icshape'].apply(lambda x: replace_nan(x))

    # Convert icshape scores arrays to comma separated strings
    df['icshape'] = df['icshape'].apply(lambda x: ",".join([str(el) for el in x]))

    # Uniform to PrismNet template type | name | seq | icshape | score | label
    # Drop columns
    df.drop(['chr', 'start_pos', 'end_pos', 'icshape_cov'], axis=1, inplace=True)

    # Move 'score' and 'label' to end
    df = df[['strand', 'name', 'seq', 'icshape', 'score', 'label']]

    # Convert strand to type: + -> A, - -> B
    map = {
        '+': 'A',
        '-': 'B'
    }
    df['strand'] = df['strand'].apply(lambda x: map[x])
    df['label'] = df['label'].astype(int)

    df.reset_index(drop=True, inplace=True)

    print(df.head())

    df.to_csv(args.output_tsv, sep="\t", header=None, index=False)