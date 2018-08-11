import numpy as np
import math

def sigmoid(x):
    return map(lambda y: 1 / (1+math.exp(-y)), x)

def hamming_distance(gt, est):
    if sum(gt) == 0:
        ret = -1
    else:
        ret = sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))
    return ret

def check_accuracy(net, num_batches, batch_size = 128):
    acc_gender = 0.0
    acc_age = 0.0
    cnt_g = 0
    cnt_a = 0

    # with open('./gt_age_list.txt', 'w') as f:

    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['fc_class'].data

        gts_gender = gts[:,0:2]
        gts_age = gts[:,2::]
        ests_gender = ests[:,0:2]
        ests_age = ests[:,2::]

        rows = len(ests)
        ests_gf = [[0]*2 for _ in range(rows)]
        ests_af = [[0]*8 for _ in range(rows)]
        cnt = 0
        for est_gender, est_age in zip(ests_gender, ests_age):
            est_gender = est_gender.tolist()
            est_age = est_age.tolist()
            gender_id = est_gender.index(max(est_gender))
            age_id = est_age.index(max(est_age))
            ests_gf[cnt][gender_id] = 1
            ests_af[cnt][age_id] = 1
            cnt += 1


        for gt, est in zip(gts_gender, ests_gf): #for each ground truth and estimated label vector
            ret = hamming_distance(gt.astype(int), est)
            if ret == -1:
                continue
            else:
                acc_gender += ret
                cnt_g += 1

        for gt, est in zip(gts_age, ests_af): #for each ground truth and estimated label vector
            ret = hamming_distance(gt.astype(int), est)
            if ret == -1:
                continue
            else:
                acc_age += ret
                cnt_a += 1

    # return acc_gender / (num_batches * batch_size), acc_age / (num_batches * batch_size)
    print('acc_gender: %d, cnt_g: %d'%(acc_gender, cnt_g))
    print('acc_age: %d, cnt_a: %d' % (acc_age, cnt_a))
    return acc_gender / cnt_g, acc_age / cnt_a

def check_baseline_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = np.zeros((batch_size, len(gts)))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)

