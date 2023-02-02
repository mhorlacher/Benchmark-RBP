from __future__ import print_function

import tensorflow as tf

#import keras
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K

# Use tensorflow. here, cannot use .tf
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from models import residualbind
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
# from keras.callbacks import Callback
import numpy as np
import os
# import h5py
import math
import json
np.random.seed(7)  # for reproducibility
import argparse


parser = argparse.ArgumentParser(description="Training of Multi-resBind")

parser.add_argument('--train-input', type=str, help="Fasta file with train data")
parser.add_argument('--train-input-region', type=str, help="")
parser.add_argument('--output-folder-name', type=str, help="The script will create a folder with the given name and save the json file and the model .h5 file there")

args = parser.parse_args()


print("############## Arguments ##############")
print(args)
print("#######################################")

train_input = args.train_input
train_input_region = args.train_input_region
output_folder_name = args.output_folder_name





################################################################################
#
#
# create a custom loss function (weighted loss functions)
#
################################################################################

# def weighted_binary_crossentropy(data):
#     """A weighted binary crossentropy loss function
#     that works for multilabel classification
#     """
#     # create a 2 by N array with weights for 0's and 1's
#     weights = np.zeros((2, data.shape[1]))
#     # calculates weights for each label in a for loop
#     for i in range(data.shape[1]):
#         weights_n, weights_p = (data.shape[0]/(2 * (data[:,i] == 0).sum())), (data.shape[0]/(2 * (data[:,i] == 1).sum()))
#         # weights could be log-dampened to avoid extreme weights for extremly unbalanced data.
#         weights[1, i], weights[0, i] = weights_p, weights_n

#     # The below is needed to be able to work with keras' model.compile()
#     def wrapped_partial(func, *args, **kwargs):
#         partial_func = partial(func, *args, **kwargs)
#         update_wrapper(partial_func, func)
#         return partial_func

#     def wrapped_weighted_binary_crossentropy(y_true, y_pred, class_weights):
#         y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
#         # cross-entropy loss with weighting
#         out = -(y_true * K.log(y_pred)*class_weights[1] + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights[0])
#         return K.mean(out, axis=-1)

#     return wrapped_partial(wrapped_weighted_binary_crossentropy, class_weights=weights)

################################################################################
#
#
# create a custom loss function (focal loss)
#
################################################################################
def create_class_weight(labels_dict,total,mu=0.15):
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

def class_weighted_focal_loss(class_weights, gamma=2.0, class_sparsity_coefficient=10.0):
    class_weights = K.constant(class_weights, tf.float32)
    gamma = K.constant(gamma, tf.float32)
    class_sparsity_coefficient = K.constant(class_sparsity_coefficient, tf.float32)

    def focal_loss_function(y_true, y_pred):
        """
        Focal loss for multi-label classification.
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} : Ground truth labels, with shape (batch_size, number_of_classes).
            y_pred {tensor} : Model's predictions, with shape (batch_size, number_of_classes).
        Keyword Arguments:
            class_weights {list[float]} : Non-zero, positive class-weights. This is used instead
                                          of Alpha parameter.
            gamma {float} : The Gamma parameter in Focal Loss. Default value (2.0).
            class_sparsity_coefficient {float} : The weight of True labels over False labels. Useful
                                                 if True labels are sparse. Default value (1.0).
        Returns:
            loss {tensor} : A tensor of focal loss.
        """

        predictions_0 = (1.0 - y_true) * y_pred
        predictions_1 = y_true * y_pred

        cross_entropy_0 = (1.0 - y_true) * (-K.log(K.clip(1.0 - predictions_0, K.epsilon(), 1.0 - K.epsilon())))
        cross_entropy_1 = y_true *(class_sparsity_coefficient * -K.log(K.clip(predictions_1, K.epsilon(), 1.0 - K.epsilon())))

        cross_entropy = cross_entropy_1 + cross_entropy_0
        class_weighted_cross_entropy = cross_entropy * class_weights

        weight_1 = K.pow(K.clip(1.0 - predictions_1, K.epsilon(), 1.0 - K.epsilon()), gamma)
        weight_0 = K.pow(K.clip(predictions_0, K.epsilon(), 1.0 - K.epsilon()), gamma)

        weight = weight_0 + weight_1

        focal_loss_tensor = weight * class_weighted_cross_entropy

        return K.mean(focal_loss_tensor, axis=1)

    return focal_loss_function

################################################################################
# generate AsymmetricLoss(
################################################################################
def AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.5):
    gamma_neg = K.constant(gamma_neg, tf.float32)
    gamma_pos = K.constant(gamma_pos, tf.float32)

    def focal_loss_function(y_true, y_pred):
        """
        Focal loss for multi-label classification.
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} : Ground truth labels, with shape (batch_size, number_of_classes).
            y_pred {tensor} : Model's predictions, with shape (batch_size, number_of_classes).
        Keyword Arguments:
            class_weights {list[float]} : Non-zero, positive class-weights. This is used instead
                                          of Alpha parameter.
            gamma {float} : The Gamma parameter in Focal Loss. Default value (2.0).
            class_sparsity_coefficient {float} : The weight of True labels over False labels. Useful
                                                 if True labels are sparse. Default value (1.0).
        Returns:
            loss {tensor} : A tensor of focal loss.
        """

        predictions_0 = (1.0 - y_true) * y_pred
        predictions_1 = y_true * y_pred

        cross_entropy_0 = (1.0 - y_true) * (-K.log(K.clip(1.0 - predictions_0, K.epsilon(), 1.0 - K.epsilon())))
        cross_entropy_1 = y_true *(-K.log(K.clip(predictions_1, K.epsilon(), 1.0 - K.epsilon())))

        cross_entropy = cross_entropy_1 + cross_entropy_0

        weight_1 = K.pow(K.clip(1.0 - predictions_1, K.epsilon(), 1.0 - K.epsilon()), gamma_pos)
        weight_0 = K.pow(K.clip(predictions_0, K.epsilon(), 1.0 - K.epsilon()), gamma_neg)

        weight = weight_0 + weight_1

        focal_loss_tensor = weight * cross_entropy

        return K.mean(focal_loss_tensor, axis=1)

    return focal_loss_function


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

def mAP(y_true,y_pred):
	num_classes = 27
	average_precisions = []
	relevant = K.sum(K.round(K.clip(y_true, 0, 1)))
	tp_whole = K.round(K.clip(y_true * y_pred, 0, 1))
	for index in range(num_classes):
		temp = K.sum(tp_whole[:,:index+1],axis=1)
		average_precisions.append(temp * (1/(index + 1)))
	AP = Add()(average_precisions) / relevant
	mAP = K.mean(AP,axis=0)
	return mAP



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
  


################################################################################
# training Convolutional Block Attention Module
################################################################################
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
    
    x_train_region = get_region(train_input_region)
    
    # We have to create validation data by splitting because of early stopping criterion
    
    train_size = int(y_train.shape[0]*0.8)
    
    x_val = x_train[train_size:]
    
    x_val_region = x_train_region[train_size:]
    
    y_val = y_train[train_size:]
    
    x_train = x_train[:train_size]
    
    x_train_region = x_train_region[:train_size]
    
    y_train = y_train[:train_size]
    
    x_train = np.concatenate((x_train, x_train_region), axis=2)
    
    x_val = np.concatenate((x_val, x_val_region), axis=2)
    
    # Input image dimensions.
    input_shape = x_train.shape[1:]
    
    print(x_train.shape, y_train.shape)

    print(x_val.shape, y_val.shape)
    
    print(x_train_region.shape, x_val_region.shape)
    
    
    # Training parameters
    batch_size = 128
    epochs = 40
    num_task = y_train.shape[1]
    base_model = 'residualbind'
    # Choose what attention_module to use: cbam_block / se_block / None
    attention_module = None
    model_type = base_model if attention_module == None else base_model+'_'+attention_module

    print ('creating model')
    model = residualbind.ResidualBind(input_shape=input_shape,num_class=num_task)
    print ('compiling model')
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss = 'binary_crossentropy',
                  optimizer= adam,
                  metrics=['accuracy',precision,recall])
    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = output_folder_name
    model_name = 'residualbind_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # generate the class-aware weights
    total=y_train.shape[0]
    labels_dict=dict(zip(range(num_task),[sum(y_train[:,i]) for i in range(num_task)]))
    print(labels_dict)
    class_weight=create_class_weight(labels_dict,total,mu=0.5)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only= True)

    callbacks = [checkpoint]
    history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_val, y_val),
                  shuffle=True,
                  class_weight=class_weight,
                  verbose=2,
                  callbacks=callbacks)



if __name__ == "__main__":
    main()

