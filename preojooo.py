#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import numpy as np
import functools

#from thinning import thinning
from common import *

class Ojooo(Preprocessor):
    def __init__(self, fname):
        super().__init__(cv2.imread(fname))
        self._fname = fname

    def segmentate(self):
        self.reset()
        self.scale(2.0)
        self._img_orig = img_orig = self.img.copy()
        skew = self.skew(230, 255)
        if skew is None:
            print('Retry')
            skew = self.skew(20, 100)

        self.reset()
        #self.scale(2.0, cv2.INTER_NEAREST)
        self.scale(2.0, cv2.INTER_CUBIC)
        self.rotate(skew)
#        _, lev = self.levels(106, 122, 8.58)
#        th = self.hsv_threshold(lev)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
#        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        th = self.threshold(self.gray(self.hsv_levels(0, 172, 0.21, level=2)))
        closed = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('closed', closed)
        #self.scale(2.0, cv2.INTER_NEAREST)
        #self.scale(0.5, cv2.INTER_CUBIC)
        self._gray = gray = self.hsv_threshold()

        self._mask_and = mask_and = cv2.bitwise_and(255-gray, 255-gray, mask=closed)
        #cv2.imshow('255-mask_and', 255-mask_and)

        img_scale = cv2.bitwise_and(self.img, self.img, mask=mask_and) #closed)

        self._cnts, self._img_dbg = Ojooo.detect_contours(img_scale, mask_and)
        # TODO: второй проход: удаление мелких областей, отстоящих на значительное расстояние (>w) от соседних групп

    @staticmethod
    def extract_contour(shape, grp):
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

        img = np.zeros(shape, np.uint8)
        for c in grp:
            #cv2.rectangle(img_dbg,c.top,c.bot,colors[i],2)
            img[c.y:c.hy, c.x:c.wx] = c.img
#            print(c.x, c.y, c.w, c.h, c.area)

        _,contours,_ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, 4)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        y0, x0 = shape
        w0 = h0 = 0

        for cnt in contours:
            [x,y,w,h] = cv2.boundingRect(cnt)
            x0 = min(x0, x)
            y0 = min(y0, y)
            w0 = max(w0, x+w)
            h0 = max(h0, y+h)

        return img[y0:h0, x0:w0]

    @staticmethod
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
#                print('{},{} {}x{}'.format(x,y,w,h))

#                cv2.drawContours(img_dbg, [approx], -1, (0, 255, 0), 1)
#                cv2.imshow('norm',img_rgb)
#                key = cv2.waitKey(0)
                if x > 20: #h > 25 and x > 20:
                    cv2.rectangle(img_dbg,(x,y),(x+w,y+h),(0,0,255),2)
#                    roi = img_scale[y:y+h,x:x+w]
                    roi = mask[y:y+h,x:x+w]
#                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#                    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
#                    roi = cv2.erode(roi, None, iterations = 2)
#                    roismall = cv2.resize(roi,(10,10))
#                    cv2.imshow('norm3',img_dbg)
#                    key = cv2.waitKey(0)

#                    if key == 27:  # (escape to quit)
#                        sys.exit()
                    preresponses.append([x, y, w, roi, '_'])
                    cnts.append([x, y, w, h, area])
                    rects.append(Rect((x,y,w,h), area, cnt, roi))
                    maxy = max(maxy, y+h)
                    miny = min(miny, y)
#                    elif key in keys:
#                        responses.append(int(chr(key)))
#                        sample = roismall.reshape((1,100))
#                        samples = np.append(samples,sample,0)

        miny -= 2
        responses = []
        for r in preresponses:
            continue
#            cv2.rectangle(img_dbg,(r[0],miny),(r[0]+r[2],maxy),(255,0,0),2)
#            continue
            p = Preprocessor(r[3])
            im = r[3]
            p.blur()
#            cv2.imshow('norm', p.img)
#            key = cv2.waitKey(0)
#            im = pre_levels(im, 0, 170)
#            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            th = p.threshold()
#            kernel = np.ones((2,2), np.uint8)
#            im = cv2.erode(im, kernel, iterations=3)

            _,contours,_ = cv2.findContours(th.copy(), cv2.RETR_LIST, 4)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)
            found = False
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                area = cv2.contourArea(cnt)
#                print(cv2.contourArea(cnt))
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
#                        roi = img_dbg[r[1]:(r[1]+y+h),r[0]:(r[0]+x+w0)]
#                        cv2.drawContours(roi, [approx], -1, (0, 255, 0), 2)
#                        img_dbg[r[1]:(r[1]+y+h),r[0]:(r[0]+x+w0)] = roi
                        cv2.rectangle(img_dbg,(x,miny),(x+w,maxy),(0,255,0),2)
                        roi = img_scale[miny:maxy,x:x+w]
                        responses.append([x, miny, w, roi, '_'])
                        cnts.append([x, miny, w, h, area])
                        rects.append(Rect((x,miny,w,h), area, cnt, hsv_img))
#                        rsp = responses[-1]
#                        cv2.imshow('norm', cut(img_scale.copy(), (rsp[0]-15, miny-10, rsp[2]+30, maxy-miny+20)))
#                        key = cv2.waitKey(0)
                        found = True
            if not found:
                cv2.rectangle(img_dbg,(r[0],miny),(r[0]+r[2],maxy),(255,0,0),2)
                roi = img_scale[miny:maxy,r[0]:r[0]+r[2]]
                responses.append([r[0], miny, r[2], roi, '_'])
                cnts.append([r[0], miny, r[2], maxy-miny, r[4]])
#                rects.append(Rect((x,y,w,h), area, cnt))

#                rsp = responses[-1]
#                cv2.imshow('norm', cut(img_scale.copy(), (rsp[0]-10, miny-10, rsp[2]+20, maxy-miny+20)))
#                key = cv2.waitKey(0)
#            cv2.imshow('norm', img_scale[(miny-5):(maxy+5),(rsp[0]-5):(rsp[0]+rsp[2]+5)])
#            cv2.imshow('norm', im)
#            key = cv2.waitKey(0)
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
#        cnts = sorted(cnts, key=functools.cmp_to_key(cmp_xx))
        reduced = []
        for nxt in rects:
            if not len(reduced):
                reduced.append(Group(nxt))
                continue
            if reduced[-1].ispart(nxt):
#                print(len(reduced), ':', nxt.x, nxt.area, reduced[-1].last.h/nxt.h)
                reduced[-1] << nxt
            else:
                print(len(reduced), ':', nxt.x, nxt.area, reduced[-1].last.h/nxt.h)
                reduced.append(Group(nxt))

        reduced2 = []
        i = 0
        t = len(reduced)-1
        while i <= t:
            cur = reduced[i]
#            if i != t:
#                nxt = reduced[i+1]
#                x0 = max(cur, key=lambda x: x.wx)
#                x1 = max(nxt, key=lambda x: x.wx)
#                print(i, abs(x0.wx - x1.x))
            i += 1

        i = 0
        for r in reduced:
            r.img = Ojooo.extract_contour((cols,rows), r)
#            cv2.imshow('norm', r.img)
#            key = cv2.waitKey(0)
#            if key == 27: sys.exit(1)
            i += 1
        return reduced, img_dbg

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    gb = Ojooo(sys.argv[1])
    gb.segmentate()
    gb.showdbg()
    #gb.dump('out')
