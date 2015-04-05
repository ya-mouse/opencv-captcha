#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
import functools

#from thinning import thinning
from common import *

if len(sys.argv) < 2:
    sys.exit(1)

img_rgb = cv2.imread(sys.argv[1])

pre = Preprocessor(img_rgb)
pre.scale(2.0)
img_orig = pre.img.copy()
skew = pre.skew(230, 255)
if skew is None:
    print('Retry')
    skew = pre.skew(20, 100)

pre.reset()
#pre.scale(2.0, cv2.INTER_NEAREST)
pre.scale(2.0, cv2.INTER_CUBIC)
pre.rotate(skew)
_, lev = pre.levels(106, 122, 8.58)
th = pre.hsv_threshold(lev)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
th = pre.threshold(pre.gray(pre.hsv_levels(0, 172, 0.21, level=2)))
closed = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('closed', closed)
#pre.scale(2.0, cv2.INTER_NEAREST)
#pre.scale(0.5, cv2.INTER_CUBIC)
gray = pre.hsv_threshold()

#ret,img_mask = cv2.threshold(cv2.cvtColor(img_rgb2, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#ret,img_mask = cv2.threshold(pre_blur(img_rgb2)[1],0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#edges2 = cv2.Canny(cv2.cvtColor(cv2.resize(img_rgb2, (0, 0), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY),50,150,apertureSize = 3)
edges2 = cv2.Canny(gray,50,150,apertureSize = 3)
#edges2 = cv2.Canny(cv2.cvtColor(img_rgb2, cv2.COLOR_BGR2GRAY),50,150,apertureSize = 3)
#img_mask = cv2.adaptiveThreshold(cv2.cvtColor(img_rgb2, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY_INV,11,2)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
#img_rgb2 = cv2.morphologyEx(img_rgb2, cv2.MORPH_CLOSE, kernel)
#edges2 = cv2.morphologyEx(edges2, cv2.MORPH_OPEN, kernel)
#edges2 = cv2.erode(edges2, kernel, iterations=1)
#img_rgb2 = cv2.cvtColor(img_rgb2, cv2.COLOR_GRAY2BGR)

# levels1: 25-200
# levels2: 0-106

l3 = cv2.cvtColor(pre.hsv_levels(170, 255, level=1), cv2.COLOR_BGR2HSV)[:,:,1]
l4 = cv2.cvtColor(pre.hsv_levels(25, 200, level=1), cv2.COLOR_BGR2HSV)[:,:,1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
l33 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

mask_and = cv2.bitwise_and(255-gray, 255-gray, mask=closed)
#cv2.imshow('255-mask_and', 255-mask_and)

def mask_by_hsv(pre, gray=None):
    if gray is None:
        gray = pre.hsv_threshold()
    gray = 255 - gray
    gray[:,:110] = 0
    gray[:,-110:] = 0
    gray[:35] = 0
    gray[-35:] = 0

    masked = cv2.bitwise_and(pre.img, pre.img, mask=gray)
    mean = cv2.mean(pre.img, mask=gray)

    color = np.uint8([[mean[:3]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h = hsv_color[0][0][0]
    print(hsv_color)
    color_lo = np.array([h-10, 100, 100])
    color_up = np.array([h+10, 255, 255])

    hsv_img = cv2.inRange(pre.hsv(), color_lo, color_up)
    masked = cv2.bitwise_and(pre.img, pre.img, mask=hsv_img)

    return masked

#cv2.imshow('aaaa', closed)
#cv2.imshow('dff', cv2.bitwise_and(pre.img, pre.img, mask=mask_and))
#cv2.imshow('im3', mask_by_hsv(pre, closed))

#cv2.imshow('l3', l3)
#cv2.imshow('l4', l4)

def cut_bg(img, cnts):
    minx = img.shape[1]
    maxy = img.shape[0]
    maxx = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 50 and area < 2500:
            [x,y,w,h] = cv2.boundingRect(cnt)
            minx = min(minx, x)
            maxx = max(maxx, x+w)

    mask_bg = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (minx-5,15,maxx+5,maxy-15)
    print(rect)
    cv2.grabCut(img,mask_bg,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask_bg==2)|(mask_bg==0),0,1).astype('uint8')
    cv2.rectangle(img,rect[:2],rect[2:],(0,0,255),2)
    return img*mask2[:,:,np.newaxis]

def cut(img, rect):
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

def detect_contours(img_scale, mask):
    _,contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 4)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    preresponses = []

    img_dbg = img_scale.copy()

    cols,rows,_ = img_scale.shape

    miny = rows
    maxy = 0
    cnts = []
    rects = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        print('A', area)
        if area >= 3: # > 15 and area < 2500:
            [x,y,w,h] = cv2.boundingRect(cnt)
#            print('{},{} {}x{}'.format(x,y,w,h))

#            cv2.drawContours(img_dbg, [approx], -1, (0, 255, 0), 1)
#            cv2.imshow('norm',img_rgb)
#            key = cv2.waitKey(0)
            if x > 20: #h > 25 and x > 20:
                cv2.rectangle(img_dbg,(x,y),(x+w,y+h),(0,0,255),2)
                roi = img_scale[y:y+h,x:x+w]
#                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#                roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
#                roi = cv2.erode(roi, None, iterations = 2)
#                roismall = cv2.resize(roi,(10,10))
#                cv2.imshow('norm3',img_dbg)
#                key = cv2.waitKey(0)

#                if key == 27:  # (escape to quit)
#                    sys.exit()
                preresponses.append([x, y, w, roi, '_'])
                cnts.append([x, y, w, h, area])
                rects.append(Rect((x,y,w,h), area, cnt))
                maxy = max(maxy, y+h)
                miny = min(miny, y)
#                elif key in keys:
#                    responses.append(int(chr(key)))
#                    sample = roismall.reshape((1,100))
#                    samples = np.append(samples,sample,0)

    miny -= 2
    responses = []
    for r in preresponses:
        continue
#        cv2.rectangle(img_dbg,(r[0],miny),(r[0]+r[2],maxy),(255,0,0),2)
#        continue
        p = Preprocessor(r[3])
        im = r[3]
        p.blur()
#        cv2.imshow('norm', p.img)
#        key = cv2.waitKey(0)
#        im = pre_levels(im, 0, 170)
#        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        th = p.threshold()
#        kernel = np.ones((2,2), np.uint8)
#        im = cv2.erode(im, kernel, iterations=3)

        _,contours,_ = cv2.findContours(th.copy(), cv2.RETR_LIST, 4)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        found = False
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            area = cv2.contourArea(cnt)
#            print(cv2.contourArea(cnt))
            if area > 60 and area < 2500:
                [x,y,w,h] = cv2.boundingRect(cnt)
                w0 = w
                if h>25:
                    if x < 8:
                        w += x
                    x = 0
                    if w + w*0.05 >= r[2]:
                        w = r[2]
                    x += r[0]
#                    roi = img_dbg[r[1]:(r[1]+y+h),r[0]:(r[0]+x+w0)]
#                    cv2.drawContours(roi, [approx], -1, (0, 255, 0), 2)
#                    img_dbg[r[1]:(r[1]+y+h),r[0]:(r[0]+x+w0)] = roi
                    cv2.rectangle(img_dbg,(x,miny),(x+w,maxy),(0,255,0),2)
                    roi = img_scale[miny:maxy,x:x+w]
                    responses.append([x, miny, w, roi, '_'])
                    cnts.append([x, miny, w, h, area])
                    rects.append(Rect((x,miny,w,h), area, cnt))
#                    rsp = responses[-1]
#                    cv2.imshow('norm', cut(img_scale.copy(), (rsp[0]-15, miny-10, rsp[2]+30, maxy-miny+20)))
#                    key = cv2.waitKey(0)
                    found = True
        if not found:
            cv2.rectangle(img_dbg,(r[0],miny),(r[0]+r[2],maxy),(255,0,0),2)
            roi = img_scale[miny:maxy,r[0]:r[0]+r[2]]
            responses.append([r[0], miny, r[2], roi, '_'])
            cnts.append([r[0], miny, r[2], maxy-miny, r[4]])
#            rects.append(Rect((x,y,w,h), area, cnt))

#            rsp = responses[-1]
#            cv2.imshow('norm', cut(img_scale.copy(), (rsp[0]-10, miny-10, rsp[2]+20, maxy-miny+20)))
#            key = cv2.waitKey(0)
#        cv2.imshow('norm', img_scale[(miny-5):(maxy+5),(rsp[0]-5):(rsp[0]+rsp[2]+5)])
#        cv2.imshow('norm', im)
#        key = cv2.waitKey(0)
    def cmp_xx(a, b):
      def _cmp_xx(a, b):
        # 100% разница по X
        if a[0]+a[2] <= b[0]:
            return -1
        if a[0] > b[0]+b[2]:
            return 1

        # Пересечение по X, считаем разницу по Y
        if a[1] < b[1]:
            a[4] = True
            return -1
        a[4] = False
        return 1

      r = _cmp_xx(a, b)
      print(a, b, r)
      return r

    rects = sorted(rects, key=lambda r: r.x)
    if not rects:
        return None, img_dbg
#    cnts = sorted(cnts, key=functools.cmp_to_key(cmp_xx))
    reduced = []
    for nxt in rects:
        if not len(reduced):
            reduced.append(Group(nxt))
            continue
        if reduced[-1].ispart(nxt):
#            print(len(reduced), ':', nxt.x, nxt.area, reduced[-1].last.h/nxt.h)
            reduced[-1] << nxt
        else:
            print(len(reduced), ':', nxt.x, nxt.area, reduced[-1].last.h/nxt.h)
            reduced.append(Group(nxt))

    colors = [
        (0,255,0),
        (0,255,255),
        (255,0,255),
        (255,0,0),
        (255,255,0),
        (127,255,0),
        (0,255,127),
        (127,255,127),
    ]

    reduced2 = []
    i = 0
    t = len(reduced)-1
    while i <= t:
        cur = reduced[i]
#        if i != t:
#            nxt = reduced[i+1]
#            x0 = max(cur, key=lambda x: x.wx)
#            x1 = max(nxt, key=lambda x: x.wx)
#            print(i, abs(x0.wx - x1.x))
        i += 1

    i = 0
    for r in reduced:
        print(i, r)
        print(r.area)
        for c in r:
            cv2.rectangle(img_dbg,c.top,c.bot,colors[i],2)
        cv2.imshow('norm', img_dbg)
#        print(c.x, c.y, c.area)
        key = cv2.waitKey(0)
        if key == 27: sys.exit(1)
        i += 1
    return cnts, img_dbg

img_scale = cv2.bitwise_and(pre.img, pre.img, mask=mask_and) #closed)

cnts, img_dbg = detect_contours(img_scale, mask_and)
# TODO: второй проход: удаление мелких областей, отстоящих на значительное расстояние (>w) от соседних групп

#print(cnts)
#for c in cnts:
#    cv2.rectangle(img_dbg,(c[0],c[1]),(c[0]+c[2],c[1]+c[3]),(0,255,0),1)
#    cv2.imshow('norm', img_dbg)
#    print(c[0], c[1], c[4])
#    key = cv2.waitKey(0)
#    if key == 27: sys.exit(1)

#img_dbg = detect_contours(img_scale, closed)
#img_dbg = detect_contours(img_scale, 255-gray)

cols,rows,_ = img_dbg.shape

img = np.ndarray((cols*5,rows,3), dtype=np.uint8)
img[:cols] = img_orig
img[cols:cols*2] = img_dbg
img[cols*2:-cols*2] = cv2.cvtColor(255-mask_and, cv2.COLOR_GRAY2BGR)
img[-cols*2:-cols] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
img[-cols:] = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)


cv2.imshow('norm2', img)
key = cv2.waitKey(0)
sys.exit(1 if key == 27 else 0)

cv2.imshow('norm', img_dbg)
key = cv2.waitKey(0)
sys.exit(0)

roi_mask = img_mask[10:-10,10:-10]
roi_rgb2 = img_rgb2[10:-10,10:-10]
roi_and = cv2.bitwise_and(roi_rgb2, roi_rgb2, mask=roi_mask)
roi_and = pre_whiteness(roi_and)
#roi_and = img_rgb2

FW = 3
ones = np.array([[255,255,255], [255,255,255], [255,255,255]])
for y in range(len(roi_and)-3):
    x = 0
    i = 0
    break
    while i < len(roi_and[0])//FW*FW - FW:
        if not (roi_mask[y,(x+i):(x+i+FW)]==ones).all():
            i += 1
            continue
        color = roi_and[y,(x+i+2)]
        fill = roi_mask[y+1,(x+i):(x+i+FW)]
        sh = roi_mask[y+2,(x+i):(x+i+FW)]
        sh2 = roi_mask[y+3,(x+i):(x+i+FW)]
        off = 1
        if (fill == ones).any():
            i += 1
            continue
        if not (sh == ones).all():
            if not (sh2 == ones).all():
                i += 1
                continue
            sh = sh2
            off = 2
        if y > 16:
            print(roi_mask[y,(x+i):(x+i+FW)], fill, sh)
            print('FND', x+i, y)
        for j in range(off):
            for k in range(3):
                if (sh[k] == [255,255,255]).all():
                    roi_and[y+j+1,(x+i+k)] = (0,0,255)
        i += 1

roi4 = cv2.resize(roi_and, (0, 0), fx=4.0, fy=4.0)
_,th4 = cv2.threshold(cv2.cvtColor(roi4, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
edges = cv2.Canny(th4,50,200,apertureSize = 5)

minLineLength = 100
maxLineGap = 20
lines = cv2.HoughLinesP(edges,20,np.pi/180,80,minLineLength=minLineLength, maxLineGap=maxLineGap)
if lines is not None:
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(roi4,(x1,y1),(x2,y2),(0,255,0),1)

#lines = cv2.HoughLines(edges,3,np.pi/180,60)
#for rho,theta in lines[0]:
#    a = np.cos(theta)
#    b = np.sin(theta)
#    if theta > 1.11: #> 1.4:
#        continue
#    print(theta)
#    x0 = a*rho
#    y0 = b*rho
##    print(x0,y0)
#    x1 = int(x0 + 120*(-b))
#    y1 = int(y0 + 120*(a))
#    x2 = int(x0 - 220*(-b))
#    y2 = int(y0 - 220*(a))

#    cv2.line(roi4,(x1,y1),(x2,y2),(0,0,255),1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
#kernel = np.ones((2,2), np.uint8)
#kernel = np.ones((4,4), np.uint8)
closed4 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel)
#closed4 = cv2.erode(roi4, kernel, iterations=5)

#cv2.imshow('n1', edges2)
#th = pre_threshold(cv2.cvtColor(pre_levels(roi_and, 0, 160), cv2.COLOR_BGR2GRAY))
#edges2 = cv2.Canny(th,50,150,apertureSize = 3)
cv2.imshow('nnn', img_thr)

#cv2.imshow('norm1', edges) #cv2.resize(cv2.resize(img_rgb2, (0, 0), fx=0.5, fy=0.5), (0,0), fx=8.0, fy=8.0))
#cv2.imshow('norm', cv2.resize(img, (0, 0), fx=2.0, fy=2.0)) #, cv2.COLOR_BGR2GRAY)))
#cv2.imshow('norm2', img_dbg) # thinning(cv2.resize(closed, (0, 0), fx=4.0, fy=4.0))) #, cv2.COLOR_BGR2GRAY)))
#cv2.imshow('n1', wat[0])
#cv2.imshow('n2', wat[1])
#cv2.imshow('n3', wat[2])
#cv2.imshow('n4', wat[3])
key = cv2.waitKey(0)
