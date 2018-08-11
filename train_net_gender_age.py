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

# these are the PASCAL classes, we'll need them later.
classes = np.asarray(['female', 'male', 'age0', 'age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'age7'])

# make sure we have the caffenet weight downloaded.
# if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#     print("Downloading pre-trained CaffeNet model...")
#     !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

workdir = './genderage_multilabel_with_datalayer'

solver = caffe.SGDSolver(osp.join(workdir, 'solver_gender_age.prototxt'))
# solver = caffe.SGDSolver(osp.join(workdir, 'dp_solver_gender_age.prototxt'))
# solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# solver.test_nets[0].share_with(solver.net)
solver.step(1)

transformer = tools.SimpleTransformer([82, 95, 121]) # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
image_index = 0 # First image in the batch.
plt.figure()
plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
plt.title('GT: {}'.format(classes[np.where(gtlist)]))
plt.axis('off')
plt.show()

max_iter = 4000
for itt in range(0, max_iter):
    if (itt % 20) == 0:
        # gender_acc, age_acc = check_accuracy(solver.test_nets[0], 318, 16)
        # gender_acc, age_acc = check_accuracy(solver.test_nets[0], 576, 32)
        #gender_acc, age_acc = check_accuracy(solver.test_nets[0], 288, 32)
        # gender_acc, age_acc = check_accuracy(solver.test_nets[0], 494, 32)
        gender_acc, age_acc = check_accuracy(solver.test_nets[0], 563, 32)
        print 'itt:{:3d}'.format((itt + 1) * 100), 'gender accuracy:{0:.4f}'.format(gender_acc), 'age accuracy:{0:.4f}'.format(age_acc)

    # if (itt % 29) == 0:
    #     print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 318))
    solver.step(100)
