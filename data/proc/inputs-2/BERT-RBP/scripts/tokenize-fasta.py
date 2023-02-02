import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('fasta')
parser.add_argument('-k', type=int, default=3)
parser.add_argument('--label', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

with open(args.output, 'w') as f_out:
    print('sequence\tlabel', file=f_out)
    
    with open(args.fasta) as f_in:
        for line in f_in:
            if line[0] == '>':
                continue
            line_strip = line.strip()
            tokens = [line_strip[i:(i+args.k)] for i in range(0, len(line_strip) - args.k + 1)]
            print(' '.join(tokens) + '\t' + str(args.label), file=f_out)


