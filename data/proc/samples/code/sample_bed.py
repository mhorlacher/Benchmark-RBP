# %%
import sys
import argparse
import random

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('beds', nargs='+')
    parser.add_argument('-n', type=int)
    args = parser.parse_args()
    
    if args.n < 1:
        exit(1)
    
    lines = []
    for bed in args.beds:
        with open(bed) as f:
            for line in f:
                lines.append(line.strip())
    
    if args.n > len(lines):
        raise ValueError(f'Can not sample {args.n} from {len(lines)} lines.')
    
    print(f'Sampling {args.n} from {len(lines)} lines.', file=sys.stderr)
    for line in random.choices(lines, k=args.n):
        print(line)
    
if __name__ == '__main__':
    main()