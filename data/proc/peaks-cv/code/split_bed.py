import argparse
import math
from pathlib import Path

def open_out_bed(prefix, suffix):
    return open(prefix + f'{suffix}.bed', 'w')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bed', metavar='<file.bed>')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-n', type=int)
    args = parser.parse_args()

    assert args.bed[-4:] == '.bed'

    suffix_ipos = math.ceil(math.log(args.n, 10))

    try:
        out_handles = [open_out_bed(args.prefix, f'{i:0{suffix_ipos}}') for i in range(args.n)]

        with open(args.bed) as f:
            for i, line in enumerate(f):
                print(line.strip(), file=out_handles[i % args.n])
    except:
        raise
    finally:
        for handle in out_handles:
            handle.close()