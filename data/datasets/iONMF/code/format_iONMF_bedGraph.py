# %%
import argparse

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bed', metavar='<file.bed>')
    args = parser.parse_args()

    with open(args.bed) as f:
        _ = f.readline()
        for line in f:
            row = line.strip().split('\t')
            chrom, start, end, protolabel = row[:4]

            if len(protolabel) == 1:
                strand = '+'
                label = protolabel
            elif len(protolabel) == 2:
                strand, label = list(protolabel)
            else:
                raise ValueError(f'Unexpected protolabel: \'{protolabel}\'')
            
            assert (int(start) + 1) == int(end)

            if label == '0':
                continue
            
            print('\t'.join([chrom, start, end, '.', label, strand]))