#!/usr/bin/env python

'''
SVM and KNearest digit recognition.

Sample loads a dataset of handwritten digits from '../data/digits.png'.
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Usage:
   digits.py
'''

# built-in modules
from multiprocessing.pool import ThreadPool

import os
import cv2

import numpy as np
from numpy.linalg import norm

# local modules
from common import clock, mosaic



SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 26
DIGITS_FN = 'out-min'
ABC = 'abcdefghijklmnopqrstuvwxyz'

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(l, idx, fn):
    print('loading from "%s" ...' % fn)
    digits = None
    for f in os.listdir(fn):
        digits_img = cv2.imread(fn+'/'+f, 0)
        if digits is None:
            digits = np.array([digits_img])
        else:
            digits = np.append(digits, [digits_img], axis=0)
#        digits.append(split2d(digits_img, (SZ, SZ)))
    labels = np.repeat(idx, len(digits))
    return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
#    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model = cv2.ml.KNearest_create()
        self.model.train(samples, layout=0, responses=responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.ml.SVM_RBF,
                            svm_type = cv2.ml.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.ml.SVM_create()
        self.model.setKernel(self.params['kernel_type'])
        self.model.setType(self.params['svm_type'])
        self.model.setC(self.params['C'])
        self.model.setGamma(self.params['gamma'])

    def train(self, samples, responses):
        self.model = cv2.ml.SVM_create()
        self.model.setKernel(self.params['kernel_type'])
        self.model.setType(self.params['svm_type'])
        self.model.setC(self.params['C'])
        self.model.setGamma(self.params['gamma'])
        self.model.train(samples, layout=0, responses=responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    print(labels)
    err = (labels != resp).mean()
    print('error: %.2f %%' % (err*100))

    confusion = np.zeros((26, 26), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print('confusion matrix:')
#    print(confusion)
    print()

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return resp, mosaic(10, vis)

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
#        bin_cells = bin[:26,:26], bin[26:,:26], bin[:26,26:], bin[26:,26:]
#        mag_cells = mag[:26,:26], mag[26:,:26], mag[:26,26:], mag[26:,26:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    print(__doc__)

    digits = None
    labels = None
    idx = 0
    for l in ABC:
        digits0, labels0 = load_digits(l, idx, DIGITS_FN+'/'+l)
        if digits is None:
            digits = digits0
            labels = labels0
        else:
            digits = np.append(digits, digits0, axis=0)
            labels = np.append(labels, labels0, axis=0)
        idx += 1

    print('preprocessing...')
    # shuffle digits
    rand = np.random.RandomState(int(np.random.rand()*100)) #321)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    digits2 = list(map(deskew, digits))
    samples = preprocess_hog(digits2)

    train_n = int(0.98*len(samples))
    cv2.imshow('test set', mosaic(10, digits[train_n:]))
    digits_train, digits_test = np.split(digits2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    print('training KNearest...')
    model = KNearest(k=4)
    model.train(samples_train, labels_train)
    resp, vis = evaluate_model(model, digits_test, samples_test, labels_test)
    print([ABC[int(c)] for c in labels_test])
    print([ABC[int(c)] for c in resp])
    cv2.imshow('KNearest test', vis)

    print('training SVM...')
    model = SVM(C=2.67, gamma=5.383)
    #model = SVM(C=4.416358054695249, gamma=3.1228367286228393) #C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    resp, vis = evaluate_model(model, digits_test, samples_test, labels_test)
    print([ABC[int(c)] for c in labels_test])
    print([ABC[int(c)] for c in resp])
    cv2.imshow('SVM test', vis)
#    print('saving SVM as "digits_svm.dat"...')
#    model.save('digits_svm.dat')

    cv2.waitKey(0)
