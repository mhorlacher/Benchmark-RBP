#################################################################################################################################################################
####################### Training DeepRiPe #######################################################################################################################
#################################################################################################################################################################


import sys
import os
import numpy as np
import h5py
import scipy.io
np.random.seed(7) # for reproducibility

import tensorflow as tf
#tf.python.control_flow_ops = tf


from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
import tensorflow.keras.backend as K

#from keras.utils.layer_utils import print_layer_shapes

from sklearn.metrics import f1_score
import math
import argparse
import json



parser = argparse.ArgumentParser(description="Data preprocessing for MultiRBP")

parser.add_argument('--train-input', type=str, help="Fasta file with train data")
parser.add_argument('--train-input-region', type=str, help="Fasta file with genomic region annotations")
parser.add_argument('--output-folder-name', type=str, help="The script will create a folder with the given name and save the json file and the model .h5 file there")

args = parser.parse_args()
print("############## Arguments ##############")
print(args)
print("#######################################")


train_input=args.train_input
train_input_region=args.train_input_region
output_folder_name=args.output_folder_name

################################################################################
# Accessry functions
################################################################################
def create_class_weight(labels_dict,total,mu=0.15):
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight


################################################################################
# Creating model
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################


### Model only using sequence data
def create_model1(num_task,input_len_l,input_len_r):
    K.clear_session()
    tf.random.set_random_seed(5005)
    # tf.set_random_seed(5005)
    left_dim=4
    right_dim=4
    num_units=50
    input_l=input_len_l
    input_r=input_len_r

    nb_f_l=[90,100]
    f_len_l=[7,7]
    p_len_l=[4,10]
    s_l=[2,5]
    nb_f_r=[90,100]
    f_len_r=[7,7]
    p_len_r=[10,10]
    s_r=[5,5]

    left_input = Input(shape=(input_l,left_dim),name="left_input")
    right_input = Input(shape=(input_r,right_dim),name="right_input")

    left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
    left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
    left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

    conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(left_drop1)
    merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
    merged_drop = Dropout(0.25)(merged_pool)
    merged_flat = Flatten()(merged_drop)

    hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
    output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
    model = Model(inputs=[left_input,right_input], outputs=output)
    print(model.summary())
    return model

### Model using both sequence data and region data
def create_model2(num_task,input_len_l,input_len_r):
    K.clear_session()
    tf.random.set_random_seed(5005)
    left_dim=4
    right_dim=4
    num_units=50
    input_l=input_len_l
    input_r=input_len_r

    nb_f_l=[90,100]
    f_len_l=[7,7]
    p_len_l=[4,10]
    s_l=[2,5]
    nb_f_r=[90,100]
    f_len_r=[7,7]
    p_len_r=[10,10]
    s_r=[5,5]

    left_input = Input(shape=(input_l,left_dim),name="left_input")
    right_input = Input(shape=(input_r,right_dim),name="right_input")

    left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
    left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
    left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

    right_conv1 = Conv1D(filters=nb_f_r[0],kernel_size=f_len_r[0], padding='valid',activation="relu",name="right_conv1")(right_input)
    right_pool1 = MaxPooling1D(pool_size=p_len_r[0], strides=s_r[0],name="right_pool1")(right_conv1)
    right_drop1 = Dropout(0.25,name="right_drop1")(right_pool1)

    merge = concatenate([left_drop1,right_drop1],name="merge",axis=-2)
    conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(merge)
    #merged_pool = MaxPooling1D(pool_size=4, strides=2)(conv_merged)
    merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
    merged_drop = Dropout(0.25)(merged_pool)
    merged_flat = Flatten()(merged_drop)
    hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
    output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
    model = Model(inputs=[left_input,right_input], outputs=output)
    print(model.summary())
    return model

### model using GRU
def create_model3(num_task,input_len_l,input_len_r):
    K.clear_session()
    tf.set_random_seed(5005)
    left_dim=4
    right_dim=4
    num_units=60
    input_l=input_len_l
    input_r=input_len_r

    nb_f_l=[90,100]
    f_len_l=[7,7]
    p_len_l=[4,10]
    s_l=[2,5]
    nb_f_r=[90,100]
    f_len_r=[7,7]
    p_len_r=[10,10]
    s_r=[5,5]

    left_input = Input(shape=(input_l,left_dim),name="left_input")
    right_input = Input(shape=(input_r,right_dim),name="right_input")


    left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
    left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
    left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

    right_conv1 = Conv1D(filters=nb_f_r[0],kernel_size=f_len_r[0], padding='valid',activation="relu",name="right_conv1")(right_input)
    right_pool1 = MaxPooling1D(pool_size=p_len_r[0], strides=s_r[0],name="right_pool1")(right_conv1)
    right_drop1 = Dropout(0.25,name="right_drop1")(right_pool1)

    merge = concatenate([left_drop1,right_drop1],name="merge",axis=-2)

    gru = Bidirectional(GRU(num_units,return_sequences=True),name="gru")(merge)
    #gru = Bidirectional(GRU(num_units),return_sequences=True,name="gru")(merged)
    flat = Flatten(name="flat")(gru)
    hidden1 = Dense(250, activation='relu',name="hidden1")(flat)
    output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
    model = Model(inputs=[left_input,right_input], outputs=output)
    print(model.summary())
    return model



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


def one_hot_encode(seq, seq_len=150):
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
    
    
    if os.path.exists(output_folder_name):
          os.system("rm -rf "+output_folder_name)

    os.mkdir(output_folder_name)
    
    print("Loading data")
    
    x_train,y_train_raw,all_rbps = get_data(train_input)

    # Create json file with indices
    rbp_list = list(set(all_rbps.split(",")))

    if "" in rbp_list: 
        rbp_list.remove("")

    json_file = dict(zip(rbp_list,[x for x in range(0,len(rbp_list))]))

    with open(output_folder_name+'/index_dict.json', 'w') as f:
        json.dump(json_file, f)

    y_train = np.array([one_hot_encode_y_values(x,len(json_file),json_file) for x in y_train_raw])
    
    #x_train_region = get_region(train_input[:-6]+".region.fasta")
    x_train_region = get_region(train_input_region)
    
    # We have to create validation data by splitting because of early stopping criterion
    
    train_size = int(y_train.shape[0]*0.8)
    
    x_val = x_train[train_size:]
    
    x_val_region = x_train_region[train_size:]
    
    y_val = y_train[train_size:]
    
    x_train = x_train[:train_size]
    
    x_train_region = x_train_region[:train_size]
    
    y_train = y_train[:train_size]
    
    print(x_train.shape, y_train.shape)

    print(x_val.shape, y_val.shape)
    
    print(x_train_region.shape, x_val_region.shape)

    model_funname="create_model2"
    filter_lengths = [4,5]
    input_len_l=150
    input_len_r=250
    num_epoch=40
    batchsize=128
    model_path=""
    num_task = len(rbp_list)

    print ('creating model')
    if isinstance(model_funname, str):
        dispatcher={'create_model1':create_model1, 'create_model2':create_model2,'create_model3':create_model3}
        try:
            model_funname=dispatcher[model_funname]
        except KeyError:
            raise ValueError('invalid input')
    model = model_funname(num_task,input_len_l,input_len_r)
    print ('compiling model')
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',precision,recall])
    checkpointer = ModelCheckpoint(filepath= output_folder_name+"/model.h5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    total=y_train.shape[0]
    labels_dict=dict(zip(range(num_task),[sum(y_train[:,i]) for i in range(num_task)]))
    class_weight=create_class_weight(labels_dict,total,mu=0.5)

    print ('fitting the model')
    history = model.fit([x_train,x_train_region], y_train, epochs=num_epoch, batch_size=batchsize,validation_data=([x_val,x_val_region],y_val), class_weight=class_weight, verbose=2, callbacks=[checkpointer,earlystopper])

if __name__ == "__main__":
    main()
    