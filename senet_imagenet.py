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

def se_module(bottom, num_out, reduce=16, params=conv_params):
    weight_filler = params[0]
    bias_filler = params[1]

    pool_global = L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)

    conv_reduce = L.Convolution(pool_global, kernel_size=1, stride=1, num_output=int(num_out/reduce), pad=0,
                                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                weight_filler=weight_filler, bias_filler=bias_filler)
    relu = L.ReLU(conv_reduce, in_place=True)

    conv_recover = L.Convolution(relu, kernel_size=1, stride=1, num_output=num_out, pad=0,
                                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                weight_filler=weight_filler, bias_filler=bias_filler)
    sigmoid = L.Sigmoid(conv_recover, in_place=True)
    scale_recalib = L.Scale(bottom, sigmoid, scale_param=dict(axis=0, bias_term=False), in_place=False)

    return pool_global, conv_reduce, relu, conv_recover, sigmoid, scale_recalib

# a group of conv, batch normalization, and relu layers.
# the default settings of kernel size, num of feature maps, stride and pad comes from the original paper.
def conv_bn_scale_relu(bottom, kernel_size=3, num_out=64, stride=1, pad=0, params=conv_params):
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out, bias_term=False,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1)],
                         weight_filler=weight_filler)
    bn = L.BatchNorm(conv, in_place=True)
    scale = L.Scale(conv, param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)], scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(conv, in_place=True)
    
    return conv, bn, scale, relu
                        
# a group of conv and batch normalization layers.
def conv_bn_scale(bottom, kernel_size=3, num_out=64, stride=1, pad=0, params=conv_params):
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out, bias_term=False,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1)],
                         weight_filler=weight_filler)
    bn = L.BatchNorm(conv, in_place=True)
    scale = L.Scale(conv, param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)], scale_param=dict(bias_term=True), in_place=True)
    
    return conv, bn, scale

# relu follows each block
def eltsum_relu(bottom1, bottom2):
    eltsum = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    relu = L.ReLU(eltsum, in_place=True)
    
    return eltsum, relu

# start making blocks. 
# most blocks have this shape.
def identity_residual(bottom, kernel_size=3, num_out=64, stride=1, pad=0):
    conv1, bn1, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size=1, num_out=num_out, stride=1, pad=0)
    conv2, bn2, scale2, relu2 = conv_bn_scale_relu(relu1, kernel_size=3, num_out=num_out, stride=stride, pad=1)
    conv3, bn3, scale3 = conv_bn_scale(relu2, kernel_size=1, num_out=num_out*4, stride=1, pad=0)

    pool_global, conv_reduce, relu_reduce, conv_recover, sigmoid_recover, scale_recalib = \
                                     se_module(scale3, num_out*4, reduce=16)

    eltsum, relu_after_sum = eltsum_relu(bottom, scale_recalib)
    
    return conv1, bn1, scale1, relu1, \
           conv2, bn2, scale2, relu2, \
           conv3, bn3, scale3, \
           pool_global, conv_reduce, relu_reduce, conv_recover, sigmoid_recover, scale_recalib, \
           eltsum, relu_after_sum

# this block is used to downsample the feature map
def project_residual(bottom, kernel_size=3, num_out=64, stride=1, pad=0):

    # branch 1: the settings is somewhat different from the original paper.
    conv_proj, bn_proj, scale_proj = conv_bn_scale(bottom, kernel_size=1, num_out=num_out*4, stride=stride, pad=0)
    # branch 2
    conv1, bn1, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size=1, num_out=num_out, stride=1, pad=0)
    conv2, bn2, scale2, relu2 = conv_bn_scale_relu(conv1, kernel_size=3, num_out=num_out, stride=stride, pad=1)
    conv3, bn3, scale3 = conv_bn_scale(relu2, kernel_size=1, num_out=num_out*4, stride=1, pad=0)

    pool_global, conv_reduce, relu_reduce, conv_recover, sigmoid_recover, scale_recalib = \
                                     se_module(scale3, num_out*4, reduce=16)


    eltsum, relu_after_sum = eltsum_relu(conv_proj, scale_recalib)
    
    return conv_proj, bn_proj, scale_proj, \
           conv1, bn1, scale1, relu1, \
           conv2, bn2, scale2, relu2, \
           conv3, bn3, scale3, \
           pool_global, conv_reduce, relu_reduce, conv_recover, sigmoid_recover, scale_recalib, \
           eltsum, relu_after_sum

def make_resnet(training_data='train_data_path', test_data='test_data_path', mean_file='mean.binaryproto', depth=50):
    
    # num_feature_maps = np.array([16, 32, 64]) # feature map size: [32, 16, 8]
    configs = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }
    block_config = configs[depth]
    num_feature_maps = [64, 128, 256, 512]
    n_stage = len(num_feature_maps)

    n = caffe.NetSpec()
    # make training data layer
    n.data, n.label = L.Data(source=training_data, backend=P.Data.LMDB, batch_size=256, ntop=2,
                                     transform_param=dict(crop_size=224, mean_file=mean_file, mirror=True),
                                     image_data_param=dict(shuffle=True), include=dict(phase=0))
    # make test data layer 
    n.test_data, n.test_label = L.Data(source=test_data, backend=P.Data.LMDB, batch_size=100, ntop=2,
                                     transform_param=dict(crop_size=224, mean_file=mean_file, mirror=False),
                                     include=dict(phase=1))
    # conv1 should accept both training and test data layers. But this is inconvenient to code in pycaffe.
    # You have to write two conv layers for them. To deal with this, I temporarily ignore the test data layer
    # and let conv1 accept the output of training data layer. Then, after making the whole prototxt, I postprocess
    # the top name of the two data layers, renaming their names to the same.

    n.conv = L.Convolution(n.data, kernel_size=7, stride=2, num_output=64,
                         pad=3, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=weight_filler, bias_filler=bias_filler)
    n.bn = L.BatchNorm(n.conv, in_place=True)
    n.scale = L.Scale(n.bn, scale_param=dict(bias_term=True), in_place=True)
    n.relu = L.ReLU(n.scale, in_place=True)


    n.max_pooling = L.Pooling(n.relu, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
    # set up a checkpoint so as to know where we get.
    checkpoint = 'n.max_pooling'

    # start making blocks.
    # num_feature_maps: the number of feature maps for each stage. Default is [16,32,64],
    #                   suggesting the network has three stages.
    # num_block_in_stage: a parameter from the original paper, telling us how many blocks there are in 
    #                   each stage.
    # depth           :
    for i in range(n_stage):
        num_map = num_feature_maps[i]
        nblocks = block_config[i]
        if (i == 0):
            stride = 1
        else:
            stride = 2
        for res in range(nblocks):
            # stage name
            stage = 'blk' + str(res + 1) + '_stg' + str(i+1)
            # use the projecting block when downsample the feature map
            if res == 0:
                make_res = 'n.' + 'conv_' + stage + '_proj,' + \
                           'n.' + 'bn_' + stage + '_proj,' + \
                           'n.' + 'scale_' + stage + '_proj,' + \
                           'n.' + 'conv_' + stage + '_a,' + \
                           'n.' + 'bn_' + stage + '_a, ' + \
                           'n.' + 'scale_' + stage + '_a, ' + \
                           'n.' + 'relu_' + stage + '_a, ' + \
                           'n.' + 'conv_' + stage + '_b, ' + \
                           'n.' + 'bn_' + stage + '_b, ' + \
                           'n.' + 'scale_' + stage + '_b, ' + \
                           'n.' + 'relu_' + stage + '_b, ' + \
                           'n.' + 'conv_' + stage + '_c, ' + \
                           'n.' + 'bn_' + stage + '_c, ' + \
                           'n.' + 'scale_' + stage + '_c, ' + \
                           'n.' + 'se_pool_' + stage + '_c, ' + \
                           'n.' + 'se_reduce_' + stage + ', ' + \
                           'n.' + 'se_relu_' + stage + ', ' + \
                           'n.' + 'se_recover_' + stage + ', ' + \
                           'n.' + 'se_sigmoid_' + stage + ', ' + \
                           'n.' + 'se_scale_' + stage + ', ' + \
                           'n.' + 'eltsum_' + stage + ', ' + \
                           'n.' + 'relu_after_sum_' + stage + \
                           ' = project_residual(' + checkpoint + ', num_out=num_map, stride=' + str(stride) + ')'
                exec(make_res)
                checkpoint = 'n.' + 'relu_after_sum_' + stage # where we get
                continue

            # most blocks have this shape
            make_res = 'n.' + 'conv_' + stage + '_a, ' + \
                       'n.' + 'bn_' + stage + '_a, ' + \
                       'n.' + 'scale_' + stage + '_a, ' + \
                       'n.' + 'relu_' + stage + '_a, ' + \
                       'n.' + 'conv_' + stage + '_b, ' + \
                       'n.' + 'bn_' + stage + '_b, ' + \
                       'n.' + 'scale_' + stage + '_b, ' + \
                       'n.' + 'relu_' + stage + '_b, ' + \
                       'n.' + 'conv_' + stage + '_c, ' + \
                       'n.' + 'bn_' + stage + '_c, ' + \
                       'n.' + 'scale_' + stage + '_c, ' + \
                       'n.' + 'se_pool_' + stage + '_d, ' + \
                       'n.' + 'se_reduce_' + stage + ', ' + \
                       'n.' + 'se_relu_' + stage + ', ' + \
                       'n.' + 'se_recover_' + stage + ', ' + \
                       'n.' + 'se_sigmoid_' + stage + ', ' + \
                       'n.' + 'se_scale_' + stage + ', ' + \
                       'n.' + 'eltsum_' + stage + ', ' + \
                       'n.' + 'relu_after_sum_' + stage + \
                       ' = identity_residual(' + checkpoint + ', num_out=num_map, stride=1)'
            exec(make_res)
            checkpoint = 'n.' + 'relu_after_sum_' + stage # where we get
            
    # add the pooling layer
    exec('n.pool_global = L.Pooling(' + checkpoint + ', pool=P.Pooling.AVE, global_pooling=True)')
    n.score = L.InnerProduct(n.pool_global, num_output=1000,
                                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                          weight_filler=dict(type='gaussian', std=0.01),
                                          bias_filler=dict(type='constant', value=0))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    n.acc = L.Accuracy(n.score, n.label)

    return n.to_proto()

#--------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    show_usage = """
This script is used to creat ResNet prototxt for ImageNet. 
Following the original paper, depth = {50, 101, 152 ,200} needs to be given, where

Usage: depth <option(s)> 
python senet_generator.py depth training_data_path test_data_path mean_file_path
Options:
    depth: 50, 101, 152, or 200.
    training_data_path: the path of training data (LEVELDB or LMDB).
    test_data_path: the path of test data (LEVELDB or LMDB).
    mean_file_path: the path of mean file for training data.

Examples: 
    python senet_imagenet.py 50 ./training_data ./test_data ./mean.binaryproto.
    """

    if len (sys.argv) > 5:
        raise RuntimeError('Usage: ' + sys.argv[0] + 'depth training_data_path test_data_path mean_file_path \n' + show_usage)

    depth = int(sys.argv[1])
    training_data_path = str(sys.argv[2])
    test_data_path = str(sys.argv[3])
    mean_file_path = str(sys.argv[4])


    proto_created = str(make_resnet(training_data_path, test_data_path, mean_file_path, depth))
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

    save_file = './se-net' + str(depth) + '_imagenet.prototxt'
    with open(save_file, 'w') as f:
        f.write(restnet_prototxt)

    print('Saved ' + save_file)
