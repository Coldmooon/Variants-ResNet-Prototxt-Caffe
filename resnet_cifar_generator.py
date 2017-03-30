# Created by coldmooon
import sys
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P

# The initialization used
weight_filler = dict(type='msra')
bias_filler = dict(type='constant', value=0)
conv_params = [weight_filler, bias_filler]

# a group of conv, batch normalization, and relu layers.
# the default settings of kernel size, num of feature maps, stride and pad comes from the original paper.
def conv_bn_scale_relu(bottom, kernel_size=3, num_out=64, stride=1, pad=0, params=conv_params):
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn_train = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=False, in_place=True, include=dict(phase=0))
    bn_test = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=True, in_place=True, include=dict(phase=1))
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(conv, in_place=True)
    
    return conv, bn_train, bn_test, scale, relu
                        
# a group of conv and batch normalization layers.
def conv_bn_scale(bottom, kernel_size=3, num_out=64, stride=1, pad=0, params=conv_params):
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn_train = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=False, in_place=True, include=dict(phase=0))
    bn_test = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=True, in_place=True, include=dict(phase=1))
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    
    return conv, bn_train, bn_test, scale

# relu follows each block
def eltsum_relu(bottom1, bottom2):
    eltsum = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    relu = L.ReLU(eltsum, in_place=True)
    
    return eltsum, relu

# start making blocks. 
# most blocks have this shape.
def identity_residual(bottom, kernel_size=3, num_out=64, stride=1, pad=0):
    conv1, bn1_train, bn1_test, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size=kernel_size, num_out=num_out, stride=stride, pad=pad)
    conv2, bn2_train, bn2_test, scale2 = conv_bn_scale(conv1, kernel_size=kernel_size, num_out=num_out, stride=stride, pad=pad)
    
    eltsum, relu_after_sum = eltsum_relu(bottom, conv2)
    
    return conv1, bn1_train, bn1_test, scale1, relu1, conv2, bn2_train, bn2_test, scale2, eltsum, relu_after_sum

# this block is used to downsample the feature map
def project_residual(bottom, kernel_size=3, num_out=64, stride=1, pad=0):
    # branch 1: the settings is somewhat different from the original paper.
    conv_proj, bn_proj_train, bn_proj_test, scale_proj = conv_bn_scale(bottom, kernel_size=3, num_out=num_out, stride=stride, pad=pad)
    # branch 2
    conv1, bn1_train, bn1_test, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size=kernel_size, num_out=num_out, stride=stride, pad=pad)
    conv2, bn2_train, bn2_test, scale2 = conv_bn_scale(conv1, kernel_size=kernel_size, num_out=num_out, stride=1, pad=pad)
    
    eltsum, relu_after_sum = eltsum_relu(conv_proj, conv2)
    
    return conv_proj, bn_proj_train, bn_proj_test, scale_proj, conv1, bn1_train, bn1_test, scale1, relu1, \
           conv2, bn2_train, bn2_test, scale2, eltsum, relu_after_sum
    
def make_resnet(training_data='cifar10_train', test_data='cifar10_test', mean_file='mean.binaryproto', num_res_in_stage=3):
    
    num_feature_maps = np.array([16, 32, 64]) # feature map size: [32, 16, 8]

    n = caffe.NetSpec()
    # make training data layer
    n.data, n.label = L.Data(source=training_data, backend=P.Data.LMDB, batch_size=128, ntop=2,
                                     transform_param=dict(crop_size=32, mean_file=mean_file, mirror=True),
                                     image_data_param=dict(shuffle=True), include=dict(phase=0))
    # make test data layer 
    n.test_data, n.test_label = L.Data(source=test_data, backend=P.Data.LMDB, batch_size=100, ntop=2,
                                     transform_param=dict(crop_size=32, mean_file=mean_file, mirror=False),
                                     include=dict(phase=1))
    # conv1 should accept both training and test data layers. But this is inconvenient to code in pycaffe.
    # You have to write two conv layers for them. To deal with this, I temporarily ignore the test data layer
    # and let conv1 accept the output of training data layer. Then, after making the whole prototxt, I postprocess
    # the top name of the two data layers, renaming their names to the same.
    n.conv1, n.bn_conv1_train, n.bn_conv1_test, n.scale_conv1, n.relu_conv1 = \
                      conv_bn_scale_relu(n.data, kernel_size=3, num_out=16, stride=1, pad=1, params=conv_params)

    # set up a checkpoint so as to know where we get.
    checkpoint = 'n.relu_conv1'

    # start making blocks.
    # num_feature_maps: the number of feature maps for each stage. Default is [16,32,64],
    #                   suggesting the network has three stages.
    # num_res_in_stage: a parameter from the original paper, telling us how many blocks there are in 
    #                   each stage.
    for num_map in num_feature_maps:
        num_map = int(num_map)
        for res in list(range(num_res_in_stage)):
            # stage name
            stage = 'map' + str(num_map) + '_' + str(res + 1) + '_'
            # use the projecting block when downsample the feature map
            if np.where(num_feature_maps == num_map)[0] >= 1 and res == 0:
                make_res = 'n.' + stage + 'conv_proj,' + \
                           'n.' + stage + 'bn_proj_train,' + \
                           'n.' + stage + 'bn_proj_test,' + \
                           'n.' + stage + 'scale_proj,' + \
                           'n.' + stage + 'conv_a,' + \
                           'n.' + stage + 'bn_a_train, ' + \
                           'n.' + stage + 'bn_a_test, ' + \
                           'n.' + stage + 'scale_a, ' + \
                           'n.' + stage + 'relu_a, ' + \
                           'n.' + stage + 'conv_b, ' + \
                           'n.' + stage + 'bn_b_train, ' + \
                           'n.' + stage + 'bn_b_test, ' + \
                           'n.' + stage + 'scale_b, ' + \
                           'n.' + stage + 'eltsum, ' + \
                           'n.' + stage + 'relu_after_sum' + \
                           ' = project_residual(' + checkpoint + ', num_out=num_map, stride=2, pad=1)'
                exec(make_res)
                checkpoint = 'n.' + stage + 'relu_after_sum' # where we get
                continue

            # most blocks have this shape
            make_res = 'n.' + stage + 'conv_a, ' + \
                       'n.' + stage + 'bn_a_train, ' + \
                       'n.' + stage + 'bn_a_test, ' + \
                       'n.' + stage + 'scale_a, ' + \
                       'n.' + stage + 'relu_a, ' + \
                       'n.' + stage + 'conv_b, ' + \
                       'n.' + stage + 'bn_b_train, ' + \
                       'n.' + stage + 'bn_b_test, ' + \
                       'n.' + stage + 'scale_b, ' + \
                       'n.' + stage + 'eltsum, ' + \
                       'n.' + stage + 'relu_after_sum' + \
                       ' = identity_residual(' + checkpoint + ', num_out=num_map, stride=1, pad=1)'
            exec(make_res)
            checkpoint = 'n.' + stage + 'relu_after_sum' # where we get
            
    # add the pooling layer
    exec('n.pool_global = L.Pooling(' + checkpoint + ', pool=P.Pooling.AVE, global_pooling=True)')
    n.score = L.InnerProduct(n.pool_global, num_output=10,
                                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                          weight_filler=dict(type='gaussian', std=0.01),
                                          bias_filler=dict(type='constant', value=0))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    n.acc = L.Accuracy(n.score, n.label)

    return n.to_proto()

#--------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    show_usage = """
This script is used to creat ResNet prototxt for Caffe. 
Following the original paper, N = {3, 5, 7 ,9} needs to be given, where
3  for 20-layer network
5  for 32-layer network
7  for 44-layer network
9  for 56-layer network
18 for 110-layer network

Usage: <option(s)> N
python resnet_generator.py training_data_path test_data_path mean_file_path N
Options:
   training_data_path: the path of training data (LEVELDB or LMDB).
       test_data_path: the path of test data (LEVELDB or LMDB).
       mean_file_path: the path of mean file for training data.
                    N: a parameter introduced by the original paper, meaning the number of repeat of residual
                       building block for each feature map size (32, 16, 8).
                       For example, N = 5 means that creat 5 residual building blocks for feature map size 32,
                       5 for feature map size 16, and 5 for feature map size 8. Besides, in each building block,
                       two weighted layers are included. So there are (5 + 5 + 5)*2 + 2 = 32 layers.
Examples: 
    python resnet_generator.py ./training_data ./test_data 5
    """

    if len (sys.argv) > 5:
        raise RuntimeError('Usage: ' + sys.argv[0] + 'training_data_path test_data_path mean_file_path N\n' + show_usage)

    training_data_path = str(sys.argv[1])
    test_data_path = str(sys.argv[2])
    mean_file_path = str(sys.argv[3])
    N =  int(sys.argv[4])

    proto_created = str(make_resnet(training_data_path, test_data_path, mean_file_path, N))
    # We want to use training data as the input of conv1 during training, and test data
    # during test. Unfortunately, in current pycaffe, this is very inconvenient to code
    # since you have to write two conv1 layers to accept the two data (one is for the top
    # name 'data_train' from the data layer at the training stage, the other is for top name
    # 'data_test' from the data layer at the test stage).
    # To address this,
    # I rename the training data and test data to the same. So only one conv1 is needed.
    proto_created = proto_created.replace('_test"', '"') # rename the top name 'data_test' to 'data'
    proto_created = proto_created.replace('_train"', '"') # rename the top name 'data_train' to 'data'
    restnet_prototxt = proto_created.replace('test_', '')

    save_file = './resnet' + str(6 * N + 2) + '_relu_msra_cifar10.prototxt'
    with open(save_file, 'w') as f:
        f.write(restnet_prototxt)

    print('Saved ' + save_file)
