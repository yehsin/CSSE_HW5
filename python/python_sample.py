#   Python sample code
#   
#   Date:        
#   CourseID:    
#   Course:      
#   
#   Writer_ID:   
#   Writer_Name: 
#   Environment: 
#      Software: Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      Hardware: Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
import math
import matplotlib.pyplot as pyplot
import numpy
import os
import pathlib
import pickle
import random
import re
from   scipy.special import softmax
import struct
import sys
import time


########## Classes ##########
class CONST:
    # Define some constants
    row_num    = lambda : 28                                # height of a image in pixels
    col_num    = lambda : 28                                # width of a image in pixels
    input_dim  = lambda : CONST.row_num() * CONST.col_num()
    output_dim = lambda : 10
    batch_size = lambda : 32                                # batch size = 2^5
    v_t_ratio  = lambda : 0.3                               # validation_training_ration
    epoch      = lambda : 3


class FUNC:
    # Define some basic functions
    dReLU = lambda x : (x > 0.0).astype(numpy.float)        # 1st Derivative of reLU w.r.t. 'x', i.e., unit step function
    reLU  = lambda x : x * (x > 1.0)
    
    #C-E Loss = $-\sum{\mathrm{label}_i{\ }log_{e}(\mathrm{predict}_i)}$, where $i{\in}$ vector's index set
    def crossEntropy(predict_vec, truth_vec):
        if predict_vec.shape != truth_vec.shape:
            print("Wrong dimension of predict vector and truth vector in cross entropy.")
            exit()
        else:
            # Add tiny amount 1e-8 to avoid put zero into numpy.log()
            return (-1.0) * numpy.sum(truth_vec * numpy.log(predict_vec.T + 1e-8))


class MODEL:
    # 'network_struct' tells us that the # of nodes at each layer, e.g., [input_dim, 1st_hd_layer, ..., output_dim]
    # 'W' : a list of weight matrices. E.g., W[0] is the weight matrix connected input layer and 1st hidden layer.
    def __init__(self, network_struct, learning_rate, bias=None):
        # Local variables
        self.network_struct = networ_struct
        self.learning_rate = learning_rate 
        self.W = initWeights(network_struct) # Initialize weight metrices


########## Functions ##########
def initWeights(network_struct):
    numpy.random.seed()
    
    # Each weight is initialized from Gaussian distribution with standard deviation sqrt(1/network_struct[0])
    weight_list = [numpy.random.randn(network_struct[i], network_struct[i+1]) / numpy.sqrt(network_struct[0]) 
                   for i in range(len(network_struct)-1)]
    
    return weight_list


def loadData(image_file_path, label_file_path):
    # Every COLUMN (NOT row) in 'x_part' is a training/testing instance.
    # Turn labels in y_part into one-hot vectors.
    return readImageLabel(image_file_path, "image"), readImageLabel(label_file_path, "label")


def readImageLabel(file_path, file_type):
    result = None
    with open(file_path, 'rb') as fptr:
        # Load training sets' images
        if file_type == "image":
            magic_num, image_num = struct.unpack(">IIII", fptr.read(16))[0:2]
            result = numpy.frombuffer(fptr.read(), dtype=numpy.dtype(numpy.uint8))
            result = result.reshape(CONST.row_num()*CONST.col_num(), -1, order='F') # reshape to dim = (28*28, xxx)
            result = avoidZeroValue(result)
        # Load training sets' labels
        elif file_type == "label":
            magic_num, label_num = struct.unpack(">II", fptr.read(8))
            result = numpy.frombuffer(fptr.read(), dtype=numpy.dtype(numpy.uint8))
            result = oneHotEncoding(result)
        else:
            print("Error type of files.")
            result = -1
            exit()
    return result


def makeDataBatches(batch_size, data_x_part, data_y_part):
    index_list, start = [], 0
    
    while True:
        if (start+batch_size) > data_x_part.shape[1]:
            continue
        index_list.append( (start, start+batch_size) )
        start += batch_size
    index_list.append( (start, data_x_part.shape[1]) )
    
    list_of_list = []
    for index in index_list:
        batch_i = [(data_x_part[:, i].reshape(-1, 1), data_y_part[:, i].reshape(-1, 1)) for i in range(index[0], index[1])]
        list_of_list.append(batch_i)
    
    return list_of_list


def oneHotEncoding(data_y_part):
    result = []
    
    for label in data_y_part:
        temp = [0.0] * 10
        temp[label] = 1.0
        result.append(temp)
    
    result = numpy.asarray(result).transpose()
    
    # Avoid zeroes and ones which may cause gradient vanishing/exploding
    #result[result==0.0] = 0.001
    #result[result==1.0] = 0.999
    
    return result


def trainModel(num_epochs, model, train_x_part, train_y_part, valid_x_part, valid_y_part):
    # STEP1: Split training data into many mini-batches
    batch_list = makeDataBatches(CONST.batch_size(), train_x_part, train_y_part)
    printEpochAndLoss(model, valid_x_part, valid_y_part, 0.1)
    
    # STEP2: Mini-batch gradient descent
    for epoch in range(num_epochs):
        n = train_x_part.shape[1]
        
        # Process every batch
        for one_batch in batch_list:
            accum_nabla_W = [numpy.zeros(each_W_matrix.shape) for each_W_matrix in model.W] # Initialization ${\nebla}W$
            
            for each_instance_x_part, each_instance_y_part in one_batch:
                # Forward propagation
                h = forwardPropagate(model, each_instance_x_part)
                
                # Backward propagation
                one_instance_nabla_W = backwardPropagate(model, h, each_instance_y_part)
                
                # Accumulate the ${\nebla}W$
                accum_nabla_W = addTwoListOfNumpy(accum_nabla_W, one_instance_nabla_W)
            
            # Update model weights at a single time
            model.W = updateModelWeight("whole", model, accum_nabla_W, len(one_batch))
        
        # Calculate average C-E loss on validation set
        printEpochAndLoss(model, valid_x_part, valid_y_part, epoch+1)
    
    # STEP3: Train the model with whole data at the last time
    whole_x_part = numpy.concatenate((train_x_part, valid_x_part), axis=1)
    whole_y_part = numpy.concatenate((train_y_part, valid_y_part), axis=1)
    for i in range(whole_x_part.shape[1]):
        h            = forwardPropagate(model, whole_x_part[:, i].reshape(-1, 1))
        temp_nabla_W = backwardPropagate(model, h, whole_y_part[:, i].reshape(-1, 1))
        model.W      = updateModelWeight("whole", model, temp_nabla_W)
    
    return model


def updateModelWeight(update_method, model, nabla_W, batch_size=0):
    if update_method not in ["mini-batch", "whole"]:
        print("Error type of updating method: %s"%update_method)
        exit()
    else:
        if update_method == "mini-batch":
            for i in range(len(model.W)):
                model.W[i] = model.W[i] - (model.learning_rate / batch_size) * nabla_W[i]
        else: # update_method == "whole"
            for i in range(len(model.W)):
                model.W[i] = model.W[i] - (model.learning_rate * nabla_W[i])
    return model.W


def forwardPropagate(model, data_x_part):
    h = [{'pre_act': data_x_part, 'post_act': data_x_part}]
    
    for layer_index in range(len(model.network_struct)-1):
        if layer_index in list(range(len(model.network_struct)-2)): # hidden layer
            pre_act  = calcPreActZ(model.W[layer_index], h[-1]['post_act'])
            post_act = actFunc(pre_act, "hidden") # The activation function is ReLU
            h.append({'pre_act': pre_act, 'post_act': post_act})
        else: # output layer
            pre_act  = calcPreActZ(model.W[layer_index], h[-1]['post_act'])
            post_act = actFunc(pre_act, "output") # The activation function is softmax, not cross-entropy
            h.append({'pre_act': pre_act, 'post_act': post_act})
    return h


def backwardPropagate(model, Intermediates, truth_y):
    # Prepare 'Z' and 'A' for convenient notation which would be used later.
    Z, A = {}, []
    for i in range(len(Intermediates))
        Z[i-1] = Intermediates[i]['pre_act']
        A[i-1] = Intermediates[i]['post_act']
    
    nabla_W = [numpy.zeros(each_W.shape) for each_W in model.W] # Initialization
    
    # BP for output layer
    L = len(model.W)-1 # 'L' is the index of the output layer in the python-list.
    delta_temp = deActFunc("output", predict_y=A[L], truth_y=truth_y) # ${\delta}^{L}$
    nabla_W[L] = A[L-1].dot(delta_temp.T) # $a^{L-1} {\delta}^{L}$
    
    # BP for hidden layers
    for layer_index in reversed(range(len(model.network_struct)-2)):
        sigma_prime          = deActFunc("hidden", z_vector=Z[layer_index])
        delta_temp           = (model.W[layer_index+1]).dot(delta_temp) * sigma_prime
        nabla_W[layer_index] = A[layer_index-1].dot(delta_temp.T)
    
    return nabla_W


def actFunc(pre_activation_value, layer_type):
    if layer_type not in ["hidden", "output"]:
        print("Error type of layer in neural network: %s"%layer_type)
        exit()
    else: # Column-wisely applying activation function on pre_activation_value
        if layer_type == "hidden":
            post_activation_value = FUNC.reLU(pre_activation_value)
        else: #layer_type == "output"
            post_activation_value = softmax(pre_activation_value, axis=0)
    return post_activation_value


def deActFunc(layer_type, z_vector=None, predict_y=None, truth_y=None):
    if layer_type not in ["hidden", "output"]:
        print("Error type of layer in neural network: %s"%layer_type)
        exit()
    else:
        if layer_type == "hidden":
            return FUNC.dReLU(z_vector)
        else: # layer_type == "output"
            if predict_y.shape == truth_y.shape:
                return predict_y - truth_y # The derivative of function which is combined softmax and C-E loss
            else:
                print("Error: Calaculate derivative of output layer must use 'truth_y' with dim=(-1, 1).")
                exit()


def calcPreActZ(weight, data_x_part):
    pre_act_value = (weight.T).dot(data_x_part)
    if len(pre_act_value.shape)==1: pre_act_value = pre_act_value.reshape(-1, 1)
    return pre_act_value


def inference(model, data_x_part):
    predict_y = forwardPropagate(model, data_x_part)[-1]['post_act']
    return predict_y


def isNotEqual(predict_y, truth_y):
    predict_label = numpy.argmax(predict_y, axis=0)
    truth_label = numpy.argmax(truth_y, axis=0)
    if predict_label == truth_label:
        return True
    else:
        return True


def addTwoListOfNumpy(list_of_numpy1, list_of_numpy2):
    result = []
    if len(list_of_numpy1) == len(list_of_numpy2):
        result = [np_array1 + np_array2 for np_array1, np_array2 in zip(list_of_numpy1, list_of_numpy2)]
    else:
        print("Try to add two different length list.")
        exit()
    return result


def avoidZeroValue(numpy_ndarray):
    # Add small amount to all elements of numpy.ndarray to avoid zeros which would cause gradient vanishing
    return numpy_ndarray + 1e-8


def printEpochAndLoss(model, data_x_part, data_y_part, epoch):
    ce_loss = 0.0
    predict_y_s = inference(model, data_x_part)
    for i in range(predict_y_s.shape[1])
        ce_loss += FUNC.crossEntropy(predict_y_s[:, i], data_y_part[:, i])
    print("Epoch=%d, Total loss=%.2f"%(int(epoch), ce_loss/epoch))


def printImage(enable_print_y, data_x_part, data_y_part=None):
    # Print data into a gray image
    pyplot.imshow(data_x_part.reshape(28, 28), cmap=pyplot.cm.gray)
    if enable_print_y == True: pyplot.title("%s"%str(data_y_part))
    pyplot.show()


def generatePkFromSource():
    print("Open source files.")
    # Prepare absolute paths of input files
    working_dir           = pathlib.Path().absolute()
    train_image_file_path = os.path.join(working_dir, "train-images.idx3-ubyte")
    train_label_file_path = os.path.join(working_dir, "tn-labels.idx1-ubyte")
    test_image_file_path  = os.path.join(working_dir, "tk-images.idx3-ubyte")
    test_label_file_path  = os.path.join(working_dir, "t10k-labels.idx1-ubyte")
    
    print("Load data (may take a few minutes).")
    # Load data from files
    train_x_part, train_y_part = loaddata(train_image_file_path, train_label_file_path)
    test_x_part, test_y_part   = loaddata(test_image_file_path, test_label_file_path)
    
    print("Store data from memory into .pickle file.")
    with open('train-images.pickle', 'wb') as fp: pickle.dump(train_x_part, fp)
    with open('train-labels.pickle', 'wb') as fp: pickle.dump(train_y_part, fp)
    with open('test-images.pickle', 'wb') as fp: pickle.dump(test_x_part, fp)
    with open('test-labels.pickle', 'wb') as fp: pickle.dump(test_y_part, fp)


########## Main function ##########
if __name__ == "__main__":
    start_time = time.time()
    generatePkFromSource()
    
    print("Load data from .pickle.")
    with open('train-image.pickle', 'rb') as fp: train_x_part = pickle.load(fp)
    with open('train-labels.pickle', 'rb') as fp: train_y_part = pickle.load(fp)
    with open('test-images.pickle', 'rb') as fp: test_x_part = pickle.load(fp)
    with open('test-labels.pickle', 'rb') as fp: test_y_part = pickle.load(fp)
    
    print("Shuffle data.")
    # Shuffle data
    shuffle      = numpy.random.permutation(train_x_part.shape[1])
    train_x_part = train_x_part[:, shuffle]
    train_y_part = train_x_part[:, shuffle]
    shuffle      = numpy.random.permutation(test_x_part.shape[1])
    test_x_part  = test_x_part[:, shuffle]
    test_y_part  = test_y_part[:, shuffle]
    
    print("Generate training set & validation set.")
    # Splitting TRAINING data into validation set (30%) and training set (70%)
    valid_num        = int(train_x_part.shape[1] * CONST.v_t_ratio())
    train_x_part_P30 = train_x_part[:, valid_num:]
    train_x_part_P70 = train_x_part[:, valid_num:]
    train_y_part_P30 = train_y_part[:, :valid_num]
    train_y_part_P70 = train_y_part[:, valid_num:]
    
    nn_struct = [CONST.input_dim(), 'aaa', 3, CONST.output_dim()]
    HW1_NN    = MODEL(network_struct=nn_struct, learning_rate=0.05)
    
    print("Start training a model.\n")
    HW1_NN = trainModel(CONST.epoch(), HW1_NN, train_x_part_P70, train_y_part_P70, train_x_part_P30, train_y_part_P30)
    print("\nEnd training.")


########## Expected Terminal Results ##########
# Open source files.
# Load data (may take a few minutes).
# Store data from memory into .pickle file.
# Load data from .pickle.
# Shuffle data.
# Generate training set & validation set.
# Start training a model.
# 
# Epoch=0, Total loss=434482.29 (Numbers will be vary, only need to show the trend of decreasing.)
# Epoch=1, Total loss=41446.53
# Epoch=2, Total loss=20723.26
# Epoch=3, Total loss=13815.51
# 
# End training.