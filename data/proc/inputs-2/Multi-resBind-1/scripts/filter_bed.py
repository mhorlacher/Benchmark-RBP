import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--bed', type=str, help="Input bed file")
parser.add_argument('--filt', type=str, help="String to be filtered with (e.g. negative-1)")

args = parser.parse_args()


bed = args.bed
filt = args.filt


def main():
    
    filt_len = len(filt)
    # We keep lines only if filt is either the only thing occurring in the name column or not
    with open(bed, "r") as f:
        for line in f.readlines():
            line = line.strip()
            name = line.split("\t")[3]
            # If filt in name and some other stuff, then drop
            name_list = list(set(name.split(",")))
            if filt in name_list and len(name_list) > 1:
                continue
            else:
                print(line)



if __name__ == "__main__":
    main()