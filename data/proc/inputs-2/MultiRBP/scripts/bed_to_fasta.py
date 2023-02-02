#######################################################################################################
##################### Preprocessing for multi-task models #############################################
#######################################################################################################


import pandas as pd
import argparse
from pysam import FastaFile

parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--input-bed', type=str, help="Binned .bed file with binding sites") # Output of generate_labeled_windows_bed
parser.add_argument('--genome-fasta', type=str, help="Genome .fasta file")
parser.add_argument('--output-fasta', type=str, help="")


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


def main():
    # Load bed file
    df = pd.read_csv(args.input_bed,delimiter="\t",names=["chr","start","end","name","score","strand"])

    df = df.drop(df[df["name"]=="."].index)
    # Fix: df containing NAs - why?
    df = df.dropna()

    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    
    # First output: fasta file with all sequences, sample names and binding sites
    fasta  = FastaFile(args.genome_fasta)
    # print(fasta)
    
    df['sample_id'] = df.agg(lambda x: f"{x['chr']}:{x['start']}-{x['end']}({x['strand']})", axis=1)

    # Add the (RNA) sequence and rbp names
    sequence = []
    sites = []
    rows_to_remove = []
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
                    sites.append(row["name"])
            else: # + strand
                sequence.append(seq)
                sites.append(row["name"])

    # Remove from df lines with no correspondence to the genome fasta
    df_to_remove = df.index.isin(rows_to_remove)
    df = df[~df_to_remove].reset_index(drop=True)

    print(f"{len(rows_to_remove)} lines have been removed")
    print(len(sequence))
    print(len(sites))
    print(len(df.index))
    df = df.assign(sequence=sequence)
    df = df.assign(sites=sites)

    with open(args.output_fasta, "w") as f_out:
        for index, row in df.iterrows():
            print(">%s %s\n%s" % (row["sample_id"],row["sites"],row["sequence"]), file=f_out)

if __name__ == "__main__":
    main()
    
    

