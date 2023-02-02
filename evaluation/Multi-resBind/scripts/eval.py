#################################################################################################################################################################
####################### Testing MRB ########################################################################################################################
#################################################################################################################################################################


import sys
import numpy as np
from keras.models import  load_model
import tensorflow.keras.backend as K
# from scipy.stats import pearsonr
# from keras.regularizers import l2
import argparse
import json
import pandas as pd
# import os
# from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--test-input', type=str, help="Fasta file with test data.")
parser.add_argument('--model-file', type=str, help="Saved model from training. The script will load the json file from the same directory")
parser.add_argument('--seq-len', type=int, help="Length of the input sequence" )
parser.add_argument('--output-csv', type=str, help="")


args = parser.parse_args()

test_input=args.test_input
model_file=args.model_file
seq_len=args.seq_len
output_csv=args.output_csv


json_path = "/".join(model_file.split("/")[:-1])

# We load the json file
rbp_dict = json.load(open(json_path+"/index_dict.json"))


def one_hot_encode(seq, seq_len):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((seq_len, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array


def region_to_mat(region):
    region_len = len(region)
    region= region.replace('i','0')
    region= region.replace('c','1')
    region= region.replace('3','2')
    region= region.replace('5','3')
    region= region.replace('N','4')
    region_code = np.zeros((4,region_len), dtype='float16')
    for i in range(region_len):
        if int(region[i]) != 4:
            region_code[int(region[i]),i] = 1
        else:
            region_code[0:4,i] = np.tile(0.25,4)
    return np.transpose(region_code)

def get_region(fasta):
    x=[]
    with open(fasta) as f:
        for line in f.readlines():
            if line[0]==">":
                continue
            else:
                x.append(region_to_mat(line.split(" ")[0]))

    x = np.array(x)         
    return x

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
                    x.append(one_hot_encode(line.split(" ")[0].replace('\n', ''), seq_len))
                    y.append(y_curr)
                    all_rbps+=(rbps_curr)
                    sample_ids.append(sampleid_curr)

    x = np.array(x)
    return x,y,all_rbps,sample_ids

################################################################################
# custume metric####
################################################################################

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def main():

    # create input for model
    x_test_seq,y_test_raw,all_rbps,sample_ids = get_data(test_input)

    y_test = np.array([one_hot_encode_y_values(x,len(rbp_dict),rbp_dict) for x in y_test_raw])

    x_test_region = get_region(test_input[:-6]+".region.fasta")
    # Step necessary for Multiresbind
    x_test = np.concatenate((x_test_seq, x_test_region), axis=2)

    print(x_test_seq.shape,y_test.shape)

    print(x_test_region.shape)

    print("Loading model...")
    # load model - we need to define here the custom metrics the model was trained with
    model = load_model(model_file, custom_objects={"precision": precision, "recall": recall})

    # get model prediction
    print("Running prediction on test set (sequence + region)...")
    preds = model.predict(x_test)

    # We transpose the preds so we get lists of predictions for each RBP
    rbp_preds = preds.T
    
    # Df to save results
    df = pd.DataFrame({})
    # TODO Refactor: do this outside script, in a snakerule
    for index, rbp in enumerate(list(rbp_dict)):
        predictions = rbp_preds[index]
        dataset = test_input.split("/")[1]
        fold = test_input.split("/")[2][-1] # Take only number
        model_type = "negative-2"
        df_rbp = pd.DataFrame([["Multi-resBind", dataset, rbp, fold, model_type, sample_ids[i], x] for i,x in enumerate(predictions)])
        df = pd.concat([df, df_rbp], axis=0)

    print("Saving final results to file.")
    print(df.head())
    df.to_csv(output_csv,header=False,index=False)
        
        
if __name__ == "__main__":
    main()
