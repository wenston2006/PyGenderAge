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
voc_root = '/home/liusong/Downloads/'
pascal_root = osp.join(caffe_root, 'VOCdevkit/VOC2012')

# these are the PASCAL classes, we'll need them later.
classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

# make sure we have the caffenet weight downloaded.
# if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#     print("Downloading pre-trained CaffeNet model...")
#     !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

workdir = './pascal_multilabel_with_datalayer'

solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
# solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# solver.test_nets[0].share_with(solver.net)
solver.step(1)

for itt in range(6):
    solver.step(100)
    print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50))

print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 5823/128))