#######################################################################################################
##################### Preprocessing for multi-task models #############################################
#######################################################################################################


import pandas as pd
import argparse
from pysam import FastaFile
import pyBigWig as pbw
import numpy as np

parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--input-bed', type=str, help="Binned .bed file with binding sites") # Output of generate_labeled_windows_bed
parser.add_argument('--genome-fasta', type=str, help="Genome .fasta file")
parser.add_argument('--bw-minus', type=str, help=".bw containing genomic annotations, minus strand")
parser.add_argument('--bw-plus', type=str, help=".bw containing genomic annotations, plus strand")
parser.add_argument('--region-len', type=int, help="Length of region vector (250 for deepripe, 150 for multiresbind)")
parser.add_argument('--output-fasta', type=str, help="")
parser.add_argument('--output-region-fasta', type=str, help="")

args = parser.parse_args()

def dna2rna(seq):
    result = ""
    for b in seq.upper():
        result+= "U" if b=="T" else b
    return result

def reverse_comp(seq):
    base_dict = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'R': 'R'}
    try:
        rc = "".join(base_dict[b] for b in reversed(seq))
    except Exception as e:# KeyError as error:
        print(e)
        print("Warning: unrecognized nucleotide in genome fasta, skipping line")
        rc = None
    
    return rc


def get_region_matrix_from_bed_coords(bw_minus, bw_plus, c,start,end,strand):
    #print(c)
    #print(start)
    #print(end)
    #print(strand)

    # chrM is not part of the annotation
    if c=="chrM":
        return "".join(["N" for x in range(0,args.region_len)]) 

    # TODO Replace this check by adding a parameter to control num nts in addition 
    # to consider for the region
    if args.region_len == 250:
        start-=50
        end+=50
    #     assert end-start==250, "Incorrect sequence length for region annotation"
    # else:
    #     assert end-start==250, "Incorrect sequence length for region annotation"
    
    # Using bigwig files
    try:
        if strand == "-":
            return "".join(pd.Series(bw_minus.values(c,start,end)).replace(np.nan,4).astype(int).map({0:'i',1:'c',2:'3',3:'5',4:'N'}).values)
        else:
            return "".join(pd.Series(bw_plus.values(c,start,end)).replace(np.nan,4).astype(int).map({0:'i',1:'c',2:'3',3:'5',4:'N'}).values)
    except Exception as e:
        print(e)
        return "".join(["N"]*args.region_len)
    #return "".join(["N" for x in range(0,args.region_len)])
    
def string_only_contains_ACUG(s):
    for c in s:
        if c not in ["A","C","G","U"]:
            return False
    return True

def main():
    # Load bed file
    df = pd.read_csv(args.input_bed,delimiter="\t",names=["chr","start","end","name","score","strand"])
    df = df.drop(df[df["name"]=="."].index)
    
    # Loading bigwig files for region annotation function
    bw_minus = pbw.open(args.bw_minus)
    bw_plus = pbw.open(args.bw_plus)

    # First output: fasta file with all sequences, sample names and binding sites
    fasta  = FastaFile(args.genome_fasta)
    # print(fasta)
    
    df['sample_id'] = df.agg(lambda x: f"{x['chr']}:{x['start']}-{x['end']}({x['strand']})", axis=1)
    
    # Add the (RNA) sequence and rbp names
    sequence = []
    sites = []
    rows_to_remove = []
    region = []
    for index, row in df.iterrows():
        # print(index)
        seq = None
        try:
            seq = fasta.fetch(row["chr"], row["start"], row["end"])
        except:
            seq = None
            rows_to_remove.append(index)
            print("Warning: encountered missing values in genome fasta, skipping line")

        if seq is not None:
            seq = dna2rna(seq)
            if row["strand"]=="-":
                seq = reverse_comp(seq)
                if seq is None:
                    rows_to_remove.append(index)
                else:
                    sequence.append(seq)
                    region.append(get_region_matrix_from_bed_coords(bw_minus, bw_plus, row["chr"], row["start"], row["end"], row["strand"]))
                    sites.append(row["name"])
            else: # + strand
                sequence.append(seq)
                region.append(get_region_matrix_from_bed_coords(bw_minus, bw_plus, row["chr"], row["start"], row["end"], row["strand"]))
                sites.append(row["name"])

    # Remove from df lines with no correspondence to the genome fasta
    df_to_remove = df.index.isin(rows_to_remove)
    df = df[~df_to_remove].reset_index(drop=True)

    print(f"{len(rows_to_remove)} lines have been removed")
    # print(len(sequence))
    # print(len(sites))
    # print(len(region))
    # print(len(df.index))
    df = df.assign(sequence=sequence)
    df = df.assign(sites=sites)
    df = df.assign(region=region)
    
    # Since we found non-AGUC characters in the sequence, we will remove those lines here
    
    df["seq_non_AGUC"] = df["sequence"].apply(lambda x :string_only_contains_ACUG(x))

    df = df[df["seq_non_AGUC"]==True].reset_index()

    with open(args.output_fasta, "w") as f_out:
        for index, row in df.iterrows():
            print(">%s %s\n%s" % (row["sample_id"],row["sites"],row["sequence"]), file=f_out)
        
    # we need to create a second fasta file for the region    
    # with open(args.output_fasta[:-6]+".region.fasta", "w") as f:
    with open(args.output_region_fasta, "w") as f:
        for index, row in df.iterrows():
            f.write(">%s %s\n%s \n" % (row["sample_id"],row["sites"],row["region"]))

if __name__ == "__main__":
    main()