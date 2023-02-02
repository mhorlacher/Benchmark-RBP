# %%
import argparse
import gzip

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive')
    parser.add_argument('--negative')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open(args.output, 'w') as f_out:
        print('sequence\tlabel', file=f_out)
        
        # positive
        with open(args.positive) as f_in:
            for line in f_in:
                if line[0] == '>':
                    continue
                print(line.strip() + '\t' + str(1), file=f_out)
                
        # positive
        with open(args.negative) as f_in:
            for line in f_in:
                if line[0] == '>':
                    continue
                print(line.strip() + '\t' + str(0), file=f_out)

# %%
if __name__ == '__main__':
    main()