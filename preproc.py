#!/usr/bin/python2
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

def pre_levels(image, minv, maxv, gamma=1.0):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get Value channel
    v = img_hsv[:,:,2]

    interval = maxv - minv

    _,v2 = cv2.threshold(v, maxv, 255, cv2.THRESH_TRUNC)
    _,v2 = cv2.threshold(v2, minv, 255, cv2.THRESH_TOZERO)
    cv2.normalize(v2, v2, 0, 255, cv2.NORM_MINMAX)

    img_hsv[:,:,2] = v2
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def pre_blur(image):
    blur = cv2.GaussianBlur(image,(5,5),0)
    img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return (blur, img_gray)

def pre_threshold(image):
    ret,img_thr = cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_thr

def pre_deskew(img):
    m = cv2.moments(img)
    rows,cols = img.shape
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*cols*skew], [0, 1, 0]])
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    img = cv2.warpAffine(img,M,(cols, rows),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img

img_rgb = cv2.imread(sys.argv[1])
img_scale = cv2.resize(img_rgb, (0, 0), fx=4.0, fy=4.0)
img_lev = pre_levels(img_scale, 200, 255)
#img_lev = pre_levels(img_scale, 0, 170)
_,img_gray = pre_blur(img_lev)

mask = cv2.inRange(img_gray, 240, 255)

#img_gray = pre_threshold(pre_blur(pre_levels(img_lev, 200, 255))[1])
cols,rows = img_gray.shape

cv2.imshow('norm', 255-mask)
for m in (img_gray, 255-mask):
    contours,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, 4)
    cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    rect = cv2.minAreaRect(cnt)
    if rect[2] != -90.0:
        break

print(rect)
if rect[2] != -90.0:
    M = cv2.getRotationMatrix2D(rect[0],rect[2],1)
    img_scale = cv2.warpAffine(img_scale,M,(rows,cols))
    #img_lev = pre_levels(img_scale, 200, 255)
    img_lev = pre_levels(img_scale, 0, 170)
    img_lev = pre_levels(img_lev, 100, 255)
    _,img_gray = pre_blur(img_lev)

img_thr = pre_threshold(img_gray)

cv2.imshow('norm', img_scale)
key = cv2.waitKey(0)
