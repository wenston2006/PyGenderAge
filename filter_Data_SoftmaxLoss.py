# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path as osp
from tools_net import hamming_distance
import shutil
import os
import math

def sigmoid(x):
    return map(lambda x: 1 / (1+math.exp(-x)), x)

# display plots in this notebook
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
caffe_root = '/home/liusong/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root + 'python')
sys.path.append("/home/liusong/caffe/examples/pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("/home/liusong/caffe/examples/pycaffe") # the tools file is in this folder

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

NameCtFolder = 1

# caffe.set_mode_cpu()
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

# model_def = './genderage_multilabel_with_datalayer/genderage_net.prototxt'
model_def = '/home/liusong/caffe/models/squeezenet_model/deepId_train_test_gender_age_net.prototxt'
model_weights = './squeeze_gender_age_dp__iter_50000.caffemodel'
# model_weights = '/home/liusong/caffe/models/squeezenet_model/squeeze_gender_age_did_dataaug_0126__iter_200000.caffemodel'
# model_weights = '/home/liusong/caffe/models/squeezenet_model/squeeze_gender_age_did_dataaug_0125__iter_125000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.array([82, 95, 121])  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

total_root = '/home/liusong/Data'
# Data_root = '/home/liusong/Data/gender_age_train/faceAgeGender_adience_aligned'
# Data_root = '/home/liusong/Data/faceAgeGender_imdb_aligned'
#Data_root = '/home/liusong/Data/faceAgeGender_imdb_aligned2'
Data_root = '/home/liusong/Data/2017_gender_aligned'
# Data_root = '/home/liusong/Data'
# Data_root = '/home/liusong/Data/gender_age_train'
# imgname = osp.join(Data_root, '/female/001/000000001_000000001.jpg')
# imgname = Data_root + '/female/001/000000001_000000001.jpg'
# Filter_data_root = '/home/liusong/Data/faceAgeGender_imdb_aligned_filter'
# Filter_data_root = '/home/liusong/Data/faceAgeGender_imdb_aligned_filter2'
# Filter_data_root = '/home/liusong/Data/faceAgeGender_imdb_aligned_filter3'
# Filter_data_root = '/home/liusong/Data/faceAgeGender_imdb_aligned_filter4'
# Filter_data_root = '/home/liusong/Data/faceAgeGender_imdb_aligned_filter5'
Filter_data_root = '/home/liusong/Data/gender_age_train_filter'

# list_file = 'train_list_gender.txt'
# list_file = 'valid_list_gender.txt'
# list_file = 'train_list_age_gender.txt'
# list_file = 'valid_list_age_gender.txt'
list_file = 'train_list_gender_age.txt'
# list_file = 'valid_list_gender_age.txt'
# list_file = 'valid_list_gender_age_comb.txt'

indexlist = [line.rstrip('\n') for line in open(
    osp.join(Data_root, list_file))]

for itt in range(len(indexlist)):
    sInstance = indexlist[itt]  # Get the image index
    sInfo = sInstance.split(' ')
    GenderType = int(sInfo[1])
    index = sInfo[0]

    # imgaddr = osp.join(Data_root, index)
    imgaddr = total_root + index
    if not osp.exists(imgaddr):
        continue
    image = caffe.io.load_image(imgaddr)
    transformed_image = transformer.preprocess('data', image)
    # plt.imshow(image)
    # plt.show()

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    # output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
    ests_gender = net.blobs['fc_class1'].data
    ests_age = net.blobs['fc_class2'].data

    gts_gender = np.array([0] * 2)
    gts_gender[GenderType] = 1

    ests_gf = np.array([0] * 2)
    gender_id = ests_gender[0].argmax()
    ests_gf[gender_id] = 1
    # ests_gf[ests_gender.argmax()] = 1

    pred_equal = hamming_distance(gts_gender, ests_gf)

    ests_gender_sig = sigmoid(ests_gender[0])
    if ests_gender_sig[gender_id] < 0.6 and pred_equal == 1:
        pred_equal = 0

    #pred_age = hamming_distance(gts_age, ests_af)

    print 'predicted class is:', ests_gf.argmax()

    if pred_equal != 1:
        if NameCtFolder == 1:
            NameSplit = index.split('/',2)
        # DstImgAddr = osp.join(Filter_data_root, index)
        DstImgAddr = Filter_data_root + '/' + NameSplit[2]
        # NameS = DstImgAddr.rsplit('/' ,2)
        # DstDir = NameS[0]
        # ImageName = NameS[2]
        NameS = DstImgAddr.rsplit('/', 1)
        DstDir = NameS[0]
        ImageName = NameS[1]
        if not osp.exists(DstDir):
            os.makedirs(DstDir)
        DstImgAddr = osp.join(DstDir, ImageName)
        shutil.copy(imgaddr, DstImgAddr)
        # shutil.move(imgaddr, DstImgAddr)


