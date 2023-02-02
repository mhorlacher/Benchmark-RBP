#################################################################################################################################################################
####################### Testing MultiRBP ########################################################################################################################
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
# from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--test-input', type=str, help="Fasta file with test data.")
parser.add_argument('--model-file', type=str, help="Saved model from training. The script will load the json file from the same directory")
#parser.add_argument('--auc_file', type=str, help="We save AUC scores in this file. The folder will be generated and there will be files added with prediction scores for each RBP")
parser.add_argument('--output-csv', type=str, help="")


args = parser.parse_args()

test_input=args.test_input
model_file=args.model_file
#auc_file=args.auc_file
output_csv=args.output_csv


json_path = "/".join(model_file.split("/")[:-1])

# We load the json file
rbp_dict = json.load(open(json_path+"/index_dict.json"))


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
        for line in f.readlines():
            if line[0]==">":
                # get y values
                y_curr = line.replace('\n', '').split(" ")[1].split(",")
                rbps_curr = line.replace('\n', '').split(" ")[1]+","
                sampleid_curr = line.replace('\n', '').split(" ")[0][1:]
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
                    sample_ids.append(sampleid_curr)

    x = np.array(x)
    return x,y,all_rbps,sample_ids



"""
loads model and saves predictions for each RBP in the rbp-list to separate csv files
"""

def main():

    
    # results_dir = "/".join(output_csv.split("/")[:-1])
    
    # if os.path.exists(results_dir):
    #       os.system("rm -rf "+results_dir)

    # os.mkdir(results_dir)

    # create input for model
    x_test,y_test_raw,all_rbps,sample_ids = get_data(test_input)

    y_test = np.array([one_hot_encode_y_values(x,len(rbp_dict),rbp_dict) for x in y_test_raw])

    print(x_test.shape,y_test.shape)

    print("Loading model...")
    # load model
    model = load_model(model_file)

    # get model prediction
    print("Running prediction on test set...")
    preds = model.predict(x_test)

    # We transpose the preds so we get lists of predictions for each RBP
    rbp_preds = preds.T

    # TODO G: part commented for now, not sure whether AUC and other performance metrics 
    # will be computed all together in a second phase
    
    # Compute aucs
    # aucs = []
    # for x in range(0,len(y_test)):
    #     auc = roc_auc_score(y_test[x], preds[x])
    #     #print("AUC: ",auc)
    #     aucs.append(auc)
        
    # print("Mean auc:",np.mean(auc))
    # auc_df = pd.DataFrame([sample_ids,aucs])
    # auc_df = pd.DataFrame({})
    # auc_df.to_csv(auc_file,header=False,index=False)
    
    # Df to save results
    df = pd.DataFrame({})
    # TODO Refactor: do this outside script, in a snakerule
    for index, rbp in enumerate(list(rbp_dict)):
        predictions = rbp_preds[index]
        dataset = test_input.split("/")[1]
        fold = test_input.split("/")[2][-1] # Take only number
        model_type = "negative-2"
        df_rbp = pd.DataFrame([["MultiRBP", dataset, rbp, fold, model_type, sample_ids[i], x] for i,x in enumerate(predictions)])
        df = pd.concat([df, df_rbp], axis=0)

    print("Saving final results to file.")
    print(df.head())
    df.to_csv(output_csv,header=False,index=False)
        
        
if __name__ == "__main__":
    main()
