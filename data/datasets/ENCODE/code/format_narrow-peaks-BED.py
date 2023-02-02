import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bed', metavar='<narrow-peaks.bed>')
    args = parser.parse_args()

    with open(args.bed) as f:
        for line in f:
            row = line.strip().split('\t')
            chrom, start, end, name, score, strand, fc, p = row[:8]
            print('\t'.join([chrom, start, end, name, fc, strand]))
            