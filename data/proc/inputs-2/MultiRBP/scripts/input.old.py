#################################################################################################################################################################
####################### Creating inputs for Multi-resBind  ######################################################################################################
#################################################################################################################################################################


import pybedtools
import pandas as pd
import sys
import numpy as np
from pysam import FastaFile
import argparse
import json


parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--genome', type=str, help="Genome fasta file")
parser.add_argument('--chrom_sizes', type=str, help="Chromsizes file for bedtools commands")
parser.add_argument('--transcriptome_bedfile', type=str, help="Transcriptome bedfile for obtaining the bins")
parser.add_argument('--bin_size', type=str, help="Binsize (150 in the paper)")
parser.add_argument('--folder', help="Folder containing the datasets")
parser.add_argument('--fold', help="Fold of the datasets that we want to preprocess. E.g. fold-0 ")
parser.add_argument('--bed_file', help="Name of the bedfile containing the binding sites. Due to MultiRBP's multi-task nature, this file has to be one of the positives")
parser.add_argument('--rbp_list', help="List of RBPs which should be processed. Please provide this as a comma-separated string")
parser.add_argument('--output', type=str, help="Path to output fasta and json files")
parser.add_argument('--genepred', type=str, help="Gene prediction file for region annotation")


args = parser.parse_args()
print("############## Arguments ##############")
print(args)
print("#######################################")

genome = args.genome
chrom_sizes = args.chrom_sizes
transcriptome_bedfile = args.transcriptome_bedfile
bin_size = int(args.bin_size)
folder = args.folder
rbp_list = args.rbp_list.split(",")
output = args.output



print(f"Genome files {genome} {chrom_sizes} {transcriptome_bedfile}")
print(f"Creating {bin_size} long binding sites for rbps {rbp_list}")



def dna2rna(seq):
    result = ""
    for b in seq.upper():
        result+= "U" if b=="T" else b
    return result

def reverse_comp(seq):
    base_dict = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    return "".join(base_dict[b] for b in reversed(seq))


def create_site_id(df):
    """
    takes dataframe with chr start end as first columns
    adds column with concatenated string of those
    """
    # create unique identifier for every site based on position
    chrom = df["chr"].tolist()
    start = df["start"].tolist()
    end = df["end"].tolist()

    site_id = []

    for x in range(0,len(df)):
        site_id.append(chrom[x]+str(start[x])+str(end[x]))

    df = df.assign(site_id=site_id)
    
    return df


def get_region_matrix_from_bed_coords(c,start,end,strand,genepred):
    """
    genepred should be a refFlat gene prediction file, already loaded as a dataframe
    """

    df1 = genepred[(genepred.chr == c) & (genepred.txStart <= start) & (genepred.txEnd >= end) & (genepred.strand == strand)]
    
    # if we don't find a transcript for the given genomic coordiantes, we just output a vector of Ns
    if len(df1)==0:
        return "".join(["N" for x in range(0,(end-start))])

    # we use the longest transcript as our reference
    df1["length"]=df1["txEnd"]-df1["txStart"]

    df1 = df1.sort_values("length",ascending=False)

    transcript = df1.iloc[0]
        
    region = ""
    for pos in range(start,end):
        reg = ""

        # Now, we can use this to create the region vectors

        exStart = transcript["exonStarts"].split(",")[:-1]
        exEnd = transcript["exonEnds"].split(",")[:-1]

        in_exon = False
        for index, st in enumerate(exStart):
            if pos>=int(st) and pos<=int(exEnd[index]):
                in_exon = True

        # Part of exons?

        if in_exon:# get utr and cds coordinates from transcript
            if strand=="+":
                utr5 = transcript["txStart"],transcript["cdsStart"]
                utr3 = transcript["cdsEnd"],transcript["txEnd"]
                cds = transcript["cdsStart"],transcript["cdsEnd"]
            elif strand=="-":
                utr3 = transcript["txStart"],transcript["cdsStart"]
                utr5 = transcript["cdsEnd"],transcript["txEnd"]
                cds = transcript["cdsStart"],transcript["cdsEnd"]

            # UTRs or coding sequence?
            if pos>=utr5[0] and pos<=utr5[1]:
                reg = "5"
            elif pos>=cds[0] and pos<=cds[1]:
                reg = "c"
            elif pos>=utr3[0] and pos<=utr3[1]:
                reg = "3"
            else:
                "N"
        else:
            reg = "i"

        region += reg
  
    return region





def create_model_input():
    
    print("Binning transcriptome...")

    # load positive transcriptome
    transcriptome = pd.read_table(transcriptome_bedfile,names=["chr","start","end","name","score","strand"])

    # divide into bins of 75nts (it was 50 for deepripe)
    transcriptome_binned = []
    for index, row in transcriptome.iterrows():
        start = row["start"]
        while start+bin_size < row["end"]:
            transcriptome_binned.append([row["chr"],start,start+bin_size,row["name"],row["score"],row["strand"]])
            start += bin_size

    transcriptome = pd.DataFrame(transcriptome_binned, columns =["chr","start","end","name","score","strand"])
    t_bed = pybedtools.BedTool.from_dataframe(transcriptome)

    binding_sites = [transcriptome["chr"].tolist(),transcriptome["start"].tolist(),transcriptome["end"].tolist(),transcriptome["name"].tolist(),transcriptome["score"].tolist(),transcriptome["strand"].tolist()]
    
    """
    Loop where every protein bed file is slopped and intersected with the binned transcriptome file
    We add a binding site column to the transcriptome file with the given protein name and add the amount of binding sites to the given bin
    """
    for rbp in rbp_list:
        print("Processing ",rbp)
        peaks = folder + '/'+ rbp + "/"+args.fold+"/"+args.bed_file

        peak_size = str(int((bin_size-1)/2))

        peaks_bed = pybedtools.BedTool(peaks)
        # Slop to get correct bin size
        s_bed = peaks_bed.slop(g=chrom_sizes,l=peak_size,r=peak_size)

        # overlap all the bed files and add labels
        # left outer join with min 50% overlap
        o_bed = t_bed.intersect(b=s_bed,loj=True,f=0.50,s=True)

        overlap = pd.read_table(o_bed.fn,names=["chr","start","end","name","score","strand","chr2","start2","end2","name2","score2","strand2"])

        # create site ids for easier matching for both dfs
        overlap = create_site_id(overlap)

        # Only keep unique peaks
        overlap = overlap.drop_duplicates(subset=['site_id'])

        overlap[rbp] = overlap["start2"].apply(lambda x: 0 if x==-1 else 1)

        # first, sanity check if the coords are in the right order
        assert transcriptome["start"].tolist()==overlap["start"].tolist()

        # we save the protein column
        binding_sites.append(overlap[rbp].tolist())

    # Add columns and make dataframe
    c = ['chr', 'start', 'end', 'name', 'score', 'strand']+rbp_list

    result = pd.DataFrame(binding_sites).T
    result.columns=c

    # Delete columns with no binding sites
    result['total_bs']= result.iloc[:, 6:].sum(axis=1)
    result = result[result["total_bs"]>0]
    
    # drop "total" column
    result.drop("total_bs", axis=1, inplace=True)

    print("Done creating binding sites")
    
    print("Preprocessing model input...")
    
    df = result.reset_index(drop=True)
    
    # first output: fasta file with all sequences, region annotations, sample names and binding sites
    fasta  = FastaFile(genome)
    genepred = pd.read_csv(args.genepred,sep="\t",
                   names = ["geneName","name","chr","strand","txStart","txEnd","cdsStart","cdsEnd","exonCount","exonStarts","exonEnds"]
                  )

    df['sample_id'] = df.agg(lambda x: f"{x['chr']}:{x['start']}-{x['end']}({x['strand']})", axis=1)

    # add the (RNA) sequence and rbp names
    sequence = []
    region = []
    sites = []
    for index, row in df.iterrows():
        seq = fasta.fetch(row["chr"], row["start"], row["end"])
        seq = dna2rna(seq)
        if row["strand"]=="-":
            seq = reverse_comp(seq)
        sequence.append(seq)
        # appending region, too
        region.append(get_region_matrix_from_bed_coords(row["chr"], row["start"], row["end"], row["strand"],genepred))
        
        sublist = []
        for rbp in rbp_list:
            if row[rbp]==1:
                sublist.append(rbp)
        sites.append(sublist)
    df = df.assign(sequence=sequence)
    df = df.assign(region=region)
    df = df.assign(sites=sites)
    

    with open (args.output+".fasta",'a') as f:
        for index, row in df.iterrows():
            f.write(">%s %s\n%s \n" % (row["sample_id"],",".join(row["sites"]),row["sequence"]))
            
            
    # we need to create a second fasta file for the region
    with open (args.output+".region.fasta",'a') as f:
        for index, row in df.iterrows():
            f.write(">%s %s\n%s \n" % (row["sample_id"],",".join(row["sites"]),row["region"]))
            
            

    # second output: json file to map RBPs to index in y-vector
    json_file = dict(zip(rbp_list,[x for x in range(0,len(rbp_list))]))

    with open(args.output+'.json', 'w') as f:
        json.dump(json_file, f)
    print("Done, saving...")

def main():
    
    print("Preprocessing data...")
    create_model_input()


if __name__ == "__main__":
    main()