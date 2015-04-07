#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import os
import sys
import numpy as np
import functools

#from thinning import thinning
from common import *

class Grandbux(Preprocessor):
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

        self._closed = closed = self.threshold(self.gray(self.hsv_levels(0, 172, 0.21, level=2)))
        self._gray = gray = self.hsv_threshold()

        # levels1: 25-200
        # levels2: 0-106

        self._mask_and = mask_and = cv2.bitwise_and(255-gray, 255-gray, mask=closed)

        img_scale = cv2.bitwise_and(self.img, self.img, mask=closed)
        closed[:3] = 0
        closed[-3:] = 0
        closed[:,:3] = 0
        closed[:,-3:] = 0

        self._cnts, self._img_dbg = Grandbux.detect_contours(img_scale, closed)
        # TODO: второй проход: удаление мелких областей, отстоящих на значительное расстояние (>w) от соседних групп

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
                recta = cv2.minAreaRect(cnt)
                if h > 25:
#                    cv2.rectangle(img_dbg,(x,y),(x+w,y+h),(0,0,255),2)
                    roi = img_scale[y:y+h,x:x+w]
                    roi_mask = mask[y:cols,x:x+w]
#                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#                    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
#                    roi = cv2.erode(roi, None, iterations = 2)
#                    roismall = cv2.resize(roi,(10,10))

#                    if key == 27:  # (escape to quit)
#                        sys.exit()
                    old_mean = (0, 0, 0)
                    for part in (0,1):
                        #mean = cv2.mean(roi[:,w/2:], mask=roi_mask[:,w/2:])
                        #mean = cv2.mean(roi, mask=roi_mask)
                        if part == 0:
                            mean = cv2.mean(roi[:,-w/3:], mask=roi_mask[:,-w/3:])
                            old_mean = np.array([mean])
                        else:
                            mean = cv2.mean(roi[:,:w/3], mask=roi_mask[:,:w/3])
                            # Тот же цвет во второй части картинки, пропускаем
                            if (mean >= old_mean-20).all() and (mean <= old_mean+20).all():
                                break

                        color = np.uint8([[mean[:3]]])
                        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
                        hc = hsv_color[0][0][0]
                        print('HSV', part,hsv_color)
                        print('MEAN',part,x,w, mean, recta)
                        color_lo = np.array([hc-10, 10, 10])
                        color_up = np.array([hc+10, 255, 255])

#                        roi = img_scale[y:cols,x:x+w]
                        hsv_img = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), color_lo, color_up)
                        #hsv_img = cv2.inRange(roi, color-50, color+50)
                        masked = cv2.bitwise_and(roi, roi, mask=hsv_img)

                        def get_contour(mask):
                            _,contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 4)
                            contours = sorted(contours, key = cv2.contourArea, reverse = True)
                            for cnt in contours:
                                area = cv2.contourArea(cnt)
                                print('As', area)
                                if area < 50:
                                    continue
                                [x,y,w,h] = cv2.boundingRect(cnt)
                                if h < 5:
                                    continue
                                return x,y,w,h,area

                            return 0,0,0,0,-1

                        x0,y0,w0,h0,area0 = get_contour(hsv_img)
                        if area0 == -1:
                            continue
                        if h/h0 >= 1.4:
                            x0 = 0
                            y0 = 0
                            w0 = w
                            h0 = h
                            area0 = area
                        hsv_img = hsv_img[y0:y0+h0,x0:x0+w0]
#                        cv2.imshow('norm2', hsv_img)
#                        key = cv2.waitKey(0)
                        x0 += x
                        y0 += y
                        roi0 = img_scale[y0:y0+h0,x0:x0+w0]

                        preresponses.append([x0, y0, w0, roi0, '_'])
                        cnts.append([x0, y0, w0, h0, area0])
                        rects.append(Rect((x0,y0,w0,h0), area0, cnt, hsv_img))

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
                        rects.append(Rect((x,miny,w,h), area, cnt))
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
            if False: # reduced[-1].ispart(nxt):
#                print(len(reduced), ':', nxt.x, nxt.area, reduced[-1].last.h/nxt.h)
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
#            if i != t:
#                nxt = reduced[i+1]
#                x0 = max(cur, key=lambda x: x.wx)
#                x1 = max(nxt, key=lambda x: x.wx)
#                print(i, abs(x0.wx - x1.x))
            i += 1

        i = 0
        for r in reduced:
            print(i, r)
            print(r.area)
            for c in r:
                cv2.rectangle(img_dbg,c.top,c.bot,colors[i],2)
#            cv2.imshow('norm', img_dbg)
#            print(c.x, c.y, c.area)
#            key = cv2.waitKey(0)
#            if key == 27: sys.exit(1)
            i += 1
        return reduced, img_dbg

    def segments(self, sz):
        self.segmentate()
        digits = None
        for r in self._cnts:
            img = cv2.resize(r.img, (sz, sz))
            if digits is None:
                digits = np.array([img])
            else:
                digits = np.append(digits, [img], axis=0)
        return digits

    def showdbg(self):
        cols,rows,_ = self.img.shape

        img = np.ndarray((cols*5,rows,3), dtype=np.uint8)
        img[:cols] = self._img_orig
        img[cols:cols*2] = self._img_dbg
        img[cols*2:-cols*2] = cv2.cvtColor(255-self._mask_and, cv2.COLOR_GRAY2BGR)
        img[-cols*2:-cols] = cv2.cvtColor(self._gray, cv2.COLOR_GRAY2BGR)
        img[-cols:] = cv2.cvtColor(self._closed, cv2.COLOR_GRAY2BGR)
        cv2.imshow('SHOW', img)
        return cv2.waitKey(0)

    def dump(self, dname, sz=None):
        if len(self._cnts) != 5:
            print('WRONG: {}'.format(self._fname))
            return
        i = 0
        try: os.mkdir(dname)
        except: pass
        for r in self._cnts:
            try: os.mkdir('{}/{}'.format(dname, self._fname[i]))
            except: pass
            outname = '{}/{}/{}-{}'.format(dname, self._fname[i], i, self._fname)
            if sz is None:
                cv2.imwrite(outname, r.img)
            else:
                cv2.imwrite(outname, cv2.resize(r.img, (sz, sz)))
            i += 1

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    gb = Grandbux(sys.argv[1])
    gb.segmentate()
    gb.showdbg()
    #gb.dump('out')
