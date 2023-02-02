# %%
import argparse
import gzip

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta')
    parser.add_argument('--label', default=None)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open(args.output, 'w') as f_out:
        print('sequence\tlabel', file=f_out)
        
        # positive
        with open(args.fasta) as f_in:
            for line in f_in:
                if line[0] == '>':
                    continue
                print(line.strip() + (('\t' + args.label) if args.label else ''), file=f_out)

# %%
if __name__ == '__main__':
    main()