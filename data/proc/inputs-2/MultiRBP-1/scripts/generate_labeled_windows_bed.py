#######################################################################################################
##################### Preprocessing for multi-task models #############################################
#######################################################################################################


import pandas as pd
import subprocess
import tempfile
import argparse

parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('bed_files', nargs='+', help="List of bed files with binding sites")
parser.add_argument('--transcript-bed', type=str, help="Transcript bed file")
parser.add_argument('--bin-size', type=int, help="Bin size for transcript")
# parser.add_argument('--int-bed', type=str, help="Output file, intersection of BS and transcriptome")

args = parser.parse_args()
#print("############## Arguments ##############")
#print(args)
#print("#######################################")

bed_files = args.bed_files
transcript_bed = args.transcript_bed
bin_size = args.bin_size
#int_bed = args.int_bed

bed_files = " ".join(bed_files)

def main():
    transcriptome = pd.read_table(transcript_bed,names=["chr","start","end","name","score","strand"])

    # divide into bins of 75nts (it was 50 for deepripe)
    transcriptome_binned = []
    for index, row in transcriptome.iterrows():
        start = row["start"]
        while start+bin_size < row["end"]:
            transcriptome_binned.append([row["chr"],start,start+bin_size,row["name"],row["score"],row["strand"]])
            start += bin_size

    transcriptome = pd.DataFrame(transcriptome_binned, columns =["chr","start","end","name","score","strand"])
    # Open file to save output
    # f_out = open(int_bed, "w")

    with tempfile.NamedTemporaryFile(mode='r+') as temp:
        transcriptome.to_csv(temp.name, sep='\t', header=False,index=False)
        # Omitted left outer join
        subprocess.call("bedtools intersect -s -wa -wb -a " +temp.name+ " -b "+ bed_files + """ | awk '{print $1"\t"$2"\t"$3"\t"$11"\t"$5"\t"$6}' | sort -k1,1 -k2,2n | bedtools merge -s -d -1 -c 4,5,6 -o collapse,distinct,distinct """,shell=True)
        # subprocess.call("bedtools intersect -wa -loj -a " +temp.name+ " -b "+ bed_files + """ | awk '{print $1"\t"$2"\t"$3"\t"$11"\t"$5"\t"$6}' | sort -k1,1 -k2,2n | bedtools merge -s -d -1 -c 4,5,6 -o collapse,distinct,distinct """, stdout=f_out, shell=True)
        
if __name__ == "__main__":
    main()
    
    



















