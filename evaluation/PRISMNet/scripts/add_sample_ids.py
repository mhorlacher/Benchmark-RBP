import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument('--input_ids', type=str, help="")
parser.add_argument('--input_probs', type=str, help="")
parser.add_argument('--output_csv', type=str, help="")

args = parser.parse_args()

with open(args.input_ids) as f_in_ids, open(args.input_probs) as f_in_probs, open(args.output_csv, 'w') as f_out_csv:

    for line_id, line_probs in zip(f_in_ids, f_in_probs):
        sample_id = line_id.strip().strip(">")
        prob = line_probs.strip()
        print(f'{sample_id},{prob}', file=f_out_csv)