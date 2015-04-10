#!/usr/bin/python3

import sys
import cv2
import numpy as np
from struct import pack
from zlib import crc32
from random import randint

normal = []
upside = []

def splitimg(img):
    h, w = img.shape
    for i in range(int(w/h)):
        roi0 = img[:,i*h:(i+1)*h]
        roi1 = cv2.flip(roi0, 0, 1)
        crc0 = crc32(pack('{}B'.format(len(roi0)), *roi0[0]))
        crc1 = crc32(pack('{}B'.format(len(roi1)), *roi1[0]))
        yield i, crc0, crc1, roi0

def train(img, normal, upside):
    auto = False
    for x, crc0, crc1, roi0 in splitimg(img):
        if crc0 in normal or crc1 in normal:
            continue
        if not auto:
            cv2.imshow('roi', roi0)
            if cv2.waitKey(0) == 0x31:
                normal = np.append(normal, crc0)
                upside = np.append(upside, crc1)
            else:
                normal = np.append(normal, crc1)
                upside = np.append(upside, crc0)
                auto = True
        else:
            normal = np.append(normal, crc0)
            upside = np.append(upside, crc1)
    return normal, upside

def solve(img, normal, upside):
    unk = []
    for n, crc0, crc1, roi0 in splitimg(img):
        if crc0 in normal:
            continue
        if crc1 in normal:
            return n
        # Unknown image
        unk.append(n)
    return unk[randint(0,len(unk)-1)]

if __name__ == '__main__':
    normal = np.loadtxt('normal.txt', np.uint32)
    upside = np.loadtxt('upside.txt', np.uint32)
    img = cv2.imread(sys.argv[1], 0)
    if len(sys.argv) == 3 and sys.argv[2] == '--train':
        normal, upside = train(img, normal, upside)
        np.savetxt('normal.txt', np.unique(normal), '%u')
        np.savetxt('upside.txt', np.unique(upside), '%u')
    else:
        n = solve(img, normal, upside)
        print('SOLVE: {}'.format(n))
    sys.exit(0)
