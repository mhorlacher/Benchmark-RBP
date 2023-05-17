import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--positive-fa', required=True)
parser.add_argument('--negative-fa', required=True)
parser.add_argument('-k', type=int, default=3)
parser.add_argument('--output', required=True)
args = parser.parse_args()

with open(args.output, 'w') as f_out:
    print('sequence\tlabel', file=f_out)
    
    # positive
    with open(args.positive_fa) as f_in:
        for line in f_in:
            if line[0] == '>':
                continue
            line_strip = line.strip()
            tokens = [line_strip[i:(i+args.k)] for i in range(0, len(line_strip) - args.k + 1)]
            print(' '.join(tokens) + '\t' + str(1), file=f_out)

    # negative
    with open(args.negative_fa) as f_in:
        for line in f_in:
            if line[0] == '>':
                continue
            line_strip = line.strip()
            tokens = [line_strip[i:(i+args.k)] for i in range(0, len(line_strip) - args.k + 1)]
            print(' '.join(tokens) + '\t' + str(1), file=f_out)



