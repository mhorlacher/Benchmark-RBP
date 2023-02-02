import argparse

def parse_bed(bed):
    with open(bed) as f:
        for line in f:
            row = line.strip().split('\t')
            chrom, start, end, name, score, strand = row[:6]
            
            if strand == '+':
                row[2] = str(int(start) + 1)
            elif strand == '-':
                row[1] = str(int(end) - 1)
            else:
                raise ValueError(f"Unexpected strand '{strand}'")
            
            print('\t'.join(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bed', metavar='<file.bed>')
    args = parser.parse_args()

    parse_bed(args.bed)