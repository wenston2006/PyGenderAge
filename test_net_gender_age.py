import sys
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from copy import copy
from tools_net import check_accuracy, check_baseline_accuracy

# matplotlib inline
# plt.rcParams['figure.figsize'] = (6, 6)

caffe_root = '/home/liusong/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root + 'python')
sys.path.append("/home/liusong/caffe/examples/pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("/home/liusong/caffe/examples/pycaffe") # the tools file is in this folder

import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

import tools #this contains some tools that we need

# set data root directory, e.g:
voc_root = '/home/liusong/Data/'
# pascal_root = osp.join(voc_root, 'gender_age_train')
pascal_root = voc_root

# initialize caffe for gpu mode
caffe.set_mode_cpu()
# caffe.set_mode_gpu()
# caffe.set_device(0)

workdir = './genderage_multilabel_with_datalayer'

net = caffe.Net(caffe_deploy, caffe_modelcaffe, caffe.TEST)

transform = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transform.set_transpose('data', (2, 0, 1))
transform.set_raw_scale('data', 255)
transform.set_channel_swap('data', (2, 1, 0))

# 把加载到的图片缩放到固定的大小
net.blobs['data'].reshape(1, 2, 100, 100)

image = caffe.io.load_image('/opt/data/person/1.jpg')
transformed_image = transform.preprocess('data', image)
plt.inshow(image)

# 把警告过transform.preprocess处理过的图片加载到内存
net.blobs['data'].data[...] = transformed_image

output = net.forward()

# 因为这里仅仅测试了一张图片
# output_pro的shape中有对于1000个object相似的概率
output_pro = output['prob'][0]