import argparse

parser = argparse.ArgumentParser(description="")

# parser.add_argument('--input_tsv', type=str, help="")
parser.add_argument('--input_probs', nargs='+', help="")
parser.add_argument('--output_csv', type=str, help="")
parser.add_argument('--params', nargs='+', help="")

args = parser.parse_args()

#with open(args.input_tsv) as f_in_tsv, open(args.input_probs) as f_in_probs, open(args.output_csv, 'w') as f_out_csv:

with open(args.output_csv, 'w') as f_out_csv:
    for pred in args.input_probs:
        with open(pred) as f_in_probs:
            DATASET = args.params[0]
            NAME = args.params[1]
            FOLD = args.params[2]
            NTYPE = args.params[3]
            # sample type: either positive or negative
            s_type = pred.split('/')[-1].split('.')[0]

        #    for line_tsv, line_probs in zip(f_in_tsv, f_in_probs):
            for line_probs in f_in_probs:
        #        sample_id = line_tsv.strip().strip(">")
                pred = line_probs.strip()
                sample_id = pred.split(',')[0]
                score = pred.split(',')[1]
        #        print(f'PRISMNet,{DATASET},{NAME},{FOLD},{NTYPE},{sample_id},{score},{TYPE}', file=f_out_csv)
                print(f'PRISMNet,{DATASET},{NAME},{FOLD},{NTYPE},{sample_id},{score},{s_type}', file=f_out_csv)