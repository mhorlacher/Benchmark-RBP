#################################################################################################################################################################
####################### Testing MulitRBP ########################################################################################################################
#################################################################################################################################################################


import sys
import numpy as np
from keras.models import  load_model
from scipy.stats import pearsonr
from keras.regularizers import l2
import argparse
import json
import pandas as pd
import os


parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--test-input', type=str, help="Fasta file with test data.")
parser.add_argument('--train-folder', type=str, help="path where the model and json file were saved")
#parser.add_argument('--results-dir', type=str, help="results files are named after each rbp and saved here")
parser.add_argument('--output-csv', type=str, help="")


args = parser.parse_args()

test_input=args.test_input
train_folder=args.train_folder
#results_dir=args.results_dir
output_csv=args.output_csv


# We load the json file
rbp_dict = json.load(open(train_folder+"/index_dict.json"))


def one_hot_encode(seq, seq_len=75):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((seq_len, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array


def one_hot_encode_y_values(rbps, count, rbp_dict):
    result = [0.0 for _ in range(0,count)]
    for rbp in rbps:
        index = rbp_dict[rbp]
        result[index]=1.0
    return result

def string_only_contains_ACUG(s):
    for c in s:
        if c not in ["A","C","G","U"]:
            return False
    return True


def get_data(fasta):
    x=[]
    y=[]
    all_rbps=""
    sample_ids = []
    with open(fasta) as f:
        y_curr = ""
        rbps_curr = ""
        sample_curr = ""
        for line in f.readlines():
            if line[0]==">":
                # get y values
                y_curr = line.replace('\n', '').split(" ")[1].split(",")
                rbps_curr = line.replace('\n', '').split(" ")[1]+","
                # Samples for the output files
                sample_curr = line.replace('\n', '').split(" ")[0][1:]
            else:
                seq = line.split(" ")[0].replace('\n', '')
                # We skip lines which have Ns in them. We also skip the corresponding > line.
                #if "N" in seq:
                if not string_only_contains_ACUG(seq):
                    continue
                else:
                    # get one hot encoded sequences and append the other cached values from the > line
                    x.append(one_hot_encode(line.split(" ")[0].replace('\n', '')))
                    y.append(y_curr)
                    all_rbps+=(rbps_curr)
                    # Sample ids for testing
                    sample_ids.append(sample_curr)

    x = np.array(x)
    return x,y,all_rbps,sample_ids
"""
loads model and saves predictions for each RBP in the rbp-list to separate csv files
"""

def main():

    # if os.path.exists(results_dir):
    #       os.system("rm -rf "+results_dir)

    # os.mkdir(results_dir)

    # create input for model
    x_test,y_test_raw,all_rbps,sample_ids = get_data(test_input)

    y_test = np.array([one_hot_encode_y_values(x,len(rbp_dict),rbp_dict) for x in y_test_raw])

    print(x_test.shape,y_test.shape)

    # load model
    model = load_model(train_folder+"/model.h5")

    # get model prediction
    preds = model.predict(x_test)

    # We transpose the preds so we get lists of predictions for each RBP
    rbp_preds = preds.T

    for index, rbp in enumerate(list(rbp_dict)):
        predictions = rbp_preds[index]
        sample_type = "negative-2"
        df = pd.DataFrame([["MultiRBP",test_input,rbp,sample_ids[i],x,sample_type] for i,x in enumerate(predictions)])
        df.to_csv(results_dir+"/"+rbp+".csv",header=False,index=False)

    # Df to save results
    df = pd.DataFrame({})
    # TODO Refactor: do this outside script, in a snakerule
    for index, rbp in enumerate(list(rbp_dict)):
        predictions = rbp_preds[index]
        dataset = test_input.split("/")[1]
        fold = test_input.split("/")[2]
        sample_type = "negative-2"
        df_rbp = pd.DataFrame([["MultiRBP", dataset, rbp, fold, sample_ids[i], x, sample_type] for i,x in enumerate(predictions)])
        df = pd.concat([df, df_rbp], axis=0)

    df.to_csv(output_csv,header=False,index=False)

if __name__ == "__main__":
    main()

# TODO This can be the same eval script as multirbp
# Change script 'interface' to accept input files instead of folders, same for output