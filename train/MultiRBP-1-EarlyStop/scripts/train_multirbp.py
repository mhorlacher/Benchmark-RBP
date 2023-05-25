#################################################################################################################################################################
####################### Training MulitRBP #######################################################################################################################
#################################################################################################################################################################

import tensorflow as tf
print("Checking if GPU is available")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Input, Conv1D, MaxPooling2D, MaxPooling1D, \
    AveragePooling1D, LSTM, Dropout, Bidirectional, LeakyReLU
from keras.layers.merge import concatenate
from scipy.stats import pearsonr
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import argparse
import json
import os
import h5py
import random


parser = argparse.ArgumentParser(description="Training for MultiRBP")

parser.add_argument('--train-input', type=str, help="Fasta file with train data")
# parser.add_argument('--val-input', type=str, default=None, help="Fasta file with validation data")
parser.add_argument('--output-folder-name', type=str, help="The script will create a folder with the given name and save the json file and the model .h5 file there")

args = parser.parse_args()
print("############## Arguments ##############")
print(args)
print("#######################################")


train_input=args.train_input

# if args.val_input is not None:
#     val_input=args.val_input

output_folder_name=args.output_folder_name


# Model parameters from paper
params_dict = {
        "dropout": 0.362233801349954,
        "epochs": 78,
        #"epochs" : 5, # For debug
        "batch" : 4096,
        "regu": 5.7215002041656515e-06,
        "hidden1" : 6029,
        "hidden2" : 1168,
        "filters1" : 2376,
        "hidden_sec" : 152,
        "filters_sec" : 151,
        "leaky_alpha" : 0.23149394545024274,
        "filters_long_length" : 24,
        "filters_long" : 51
    }


def one_hot_encode(seq, seq_len=75):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((seq_len, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array


def one_hot_encode_y_values(rbps, count, json_file):
    result = [0.0 for _ in range(0,count)]
    for rbp in rbps:
        index = json_file[rbp]
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
    with open(fasta) as f:
        y_curr = ""
        rbps_curr = ""
        for line in f.readlines():
            if line[0]==">":
                # get y values
                y_curr = line.replace('\n', '').split(" ")[1].split(",")
                rbps_curr = line.replace('\n', '').split(" ")[1]+","
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

    x = np.array(x)
    return x,y,all_rbps
  

def main():
    
    os.makedirs(output_folder_name, exist_ok=True)
    
    x_train,y_train_raw,all_rbps = get_data(train_input)
    
    # x_val,y_val_raw,_ = get_data(val_input)

    # Create json file with indices
    rbp_list = list(set(all_rbps.split(",")))

    if "" in rbp_list: 
        rbp_list.remove("")

    json_file = dict(zip(rbp_list,[x for x in range(0,len(rbp_list))]))

    with open(output_folder_name+'/index_dict.json', 'w') as f:
        json.dump(json_file, f)


    y_train = np.array([one_hot_encode_y_values(x,len(json_file),json_file) for x in y_train_raw])

    # y_val = np.array([one_hot_encode_y_values(x,len(json_file),json_file) for x in y_val_raw])


    indices = [i for i in range(len(y_train))]
    #indices = np.random.shuffle(indices)
    random.shuffle(indices)

    val_size = int(y_train.shape[0]*0.2)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    print("Shape before split:")
    print(x_train.shape, y_train.shape)

    x_val = np.take(x_train, indices=val_indices, axis=0)
    y_val = np.take(y_train, indices=val_indices, axis=0)

    x_train = np.take(x_train, indices=train_indices, axis=0)
    y_train = np.take(y_train, indices=train_indices, axis=0)


    # x_train = np.random.shuffle(x_train)

    # x_val = x_train[train_size:]

    # x_val = x_train[train_size:]

    # x_train = x_train[:train_size]

    # y_val = y_train[train_size:]


    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
            
    print("Building model")
    # Model creation function
    X = Input(shape=(x_train.shape[1], 4))
    conv_kernel_long = Conv1D(params_dict["filters_long"], kernel_size=params_dict["filters_long_length"], activation='relu', use_bias=True,
                              kernel_regularizer=l2(params_dict["regu"]))(X)  # Long kernel - its purpose is to identify structure preferences
    conv_kernel_11 = Conv1D(filters=params_dict["filters1"], kernel_size=11, activation='relu', use_bias=True,
                            kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 11 nucleotides
    conv_kernel_9 = Conv1D(filters=params_dict["filters1"], kernel_size=9, activation='relu', use_bias=True,
                           kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 9 nucleotides
    conv_kernel_7 = Conv1D(filters=params_dict["filters1"], kernel_size=7, activation='relu', use_bias=True,
                           kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 7 nucleotides
    conv_kernel_5 = Conv1D(filters=params_dict["filters1"], kernel_size=5, activation='relu', use_bias=True,
                           kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 5 nucleotides
    conv_kernel_5_sec = Conv1D(filters=params_dict["filters_sec"], kernel_size=5, activation='relu', use_bias=True,
                             kernel_regularizer=l2(params_dict["regu"]))(X) # kernel of 5 nucleotides - second path

    max_pool_long = MaxPooling1D(pool_size=(74 - params_dict["filters_long_length"]))(conv_kernel_long)
    max_pool_11 = MaxPooling1D(pool_size=(65))(conv_kernel_11)
    max_pool_9 = MaxPooling1D(pool_size=(67))(conv_kernel_9)
    max_pool_7 = MaxPooling1D(pool_size=(69))(conv_kernel_7)
    max_pool_5 = MaxPooling1D(pool_size=(71))(conv_kernel_5)
    max_pool_5_sec = MaxPooling1D(pool_size=(71))(conv_kernel_5_sec)
    merge2 = concatenate([max_pool_11, max_pool_7, max_pool_long, max_pool_9, max_pool_5]) #merge first path
    fl_rel = Flatten()(merge2) #Flatten layer
    fl_sec = Flatten()(max_pool_5_sec) #Flatten layer - second path
    drop_fl_sec = Dropout(params_dict["dropout"], name="drop_fl_el")(fl_sec) #Dropout
    drop_flat = Dropout(params_dict["dropout"], name="drop_flat")(fl_rel)
    hidden_dense_sec = Dense(params_dict["hidden_sec"], activation='relu')(drop_fl_sec)
    hidden_dense_relu = Dense(params_dict["hidden1"], activation='relu')(drop_flat)  # 4096
    drop_hidden_dense_relu = Dropout(params_dict["dropout"], name="drop_hidden_dense_relu")(hidden_dense_relu)
    hidden_dense_relu1 = Dense(params_dict["hidden2"], activation='relu')(drop_hidden_dense_relu)  # 1024 best
    merge_4 = concatenate([hidden_dense_sec, hidden_dense_relu1, drop_flat, hidden_dense_relu])
    Y_1 = Dense(y_train.shape[1])(merge_4) #244 originally
    Y = LeakyReLU(alpha=params_dict["leaky_alpha"])(Y_1)
    model_func = Model(inputs=X, outputs=Y)
    model_func.compile(loss='logcosh', optimizer='adam')  # adam
    print(model_func.summary())


    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    print("Training...")
    # I included the validation data. In the paper, they just train for 78 epochs without val data
    model_func.fit(x_train, y_train, batch_size=params_dict["batch"], epochs=params_dict["epochs"], verbose=1, validation_data=(x_val, y_val), callbacks=[earlystopper])
    # model_func.fit(x_train, y_train, batch_size=params_dict["batch"], epochs=params_dict["epochs"], verbose=1, callbacks=[earlystopper])

    # Save model
    model_func.save(output_folder_name+"/MultiRBP.h5")
    print("Done,saving...")
    
    
    

if __name__ == "__main__":
    main()
    
