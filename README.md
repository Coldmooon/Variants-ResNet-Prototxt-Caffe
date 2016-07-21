# ResNet-Prototxt-for-Caffe

This script is used to creat ResNet prototxt for Caffe. Following the original paper, N = {3, 5, 7 ,9} needs to be given, where
- 3  for 20-layer network
- 5  for 32-layer network
- 7  for 44-layer network
- 9  for 56-layer network
- 18 for 110-layer network

# Usage:

```
python resnet_generator.py training_data_path test_data_path mean_file_path N
```

- training_data_path: the path of training data (LEVELDB or LMDB).
-     test_data_path: the path of test data (LEVELDB or LMDB).
-     mean_file_path: the path of mean file for training data.
-                  N: a parameter introduced by the original paper, meaning the number of repeat of residualn building block for each feature map size (32, 16, 8). For example, N = 5 means that creat 5 residual building blocks for feature map size 32, 5 for feature map size 16, and 5 for feature map size 8. Besides, in each building block, two weighted layers are included. So there are (5 + 5 + 5)*2 + 2 = 32 layers.

# Examples: 

```
python resnet_generator.py ./training_data ./test_data ./mean.binaryproto 5
```