#!/usr/bin/python3

import cv2
import sys

from common import mosaic, KNearest, SVM, OCR

from pregrandbux import Grandbux

if __name__ == '__main__':
#    model = KNearest(k=4)
    model = SVM(C=2.67, gamma=5.383)

    ABC = 'abcdefghijklmnopqrstuvwxyz'
    ocr = OCR(ABC, model)
    do_load = len(sys.argv) > 1
    if do_load:
#        ocr.train('out-min', 'grandbux_svm.dat', 1.0)
        ocr.load('grandbux_svm.dat')
        samples = ocr.preprocess_hog(Grandbux(sys.argv[1]).segments(20))
        print('SOLVE: {}'.format(''.join(ocr.labels(ocr.predict(samples)))))
    else:
        print('training SVM...')
        ocr.train('out-min', 'grandbux_svm.dat')
        cv2.imshow('SVM', mosaic(25, ocr.digits))

    cv2.waitKey(0)
