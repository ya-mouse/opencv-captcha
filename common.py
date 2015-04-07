import cv2
import os
import numpy as np
from numpy.linalg import norm
import itertools as it

class Preprocessor:
    def __init__(self, image):
        self._orig = image
        self._img = image.copy()
        self._isblack = None

    @property
    def img(self):
        return self._img

    def save(self):
        self._orig = self._img.copy()

    def reset(self):
        self._img = self._orig.copy()

    def scale(self, factor, interpolation=cv2.INTER_LINEAR):
        self._img = cv2.resize(self._img, (0,0), fx=factor, fy=factor, interpolation=interpolation)

    def levels(self, minv, maxv, gamma=1.0, img=None):
        if img is None:
            img = self._img

        interval = maxv - minv

        _ = None
        if maxv < 255:
            _,img = cv2.threshold(img, maxv, 255, cv2.THRESH_TRUNC)
        if minv > 0:
            _,img = cv2.threshold(img, minv, 255, cv2.THRESH_TOZERO)
        if _ is not None:
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        if gamma != 1.0:
            lut = np.array([i / 255.0 for i in range(256)])
            igamma = 1.0 / gamma
            lut = cv2.pow(lut, igamma) * 255.0
            abs64f = np.absolute(cv2.LUT(img, lut))
            img = np.uint8(abs64f)
        return _, img

    def hsv_levels(self, minv, maxv, gamma=1.0, img=None, level=2):
        if img is None:
            img = self._img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Get Value channel
        h = img_hsv[:,:,0]
        s = img_hsv[:,:,1]
        v = img_hsv[:,:,2]
        lev = img_hsv[:,:,level]
#        cv2.imshow('lev_h', h)
#        cv2.imshow('lev_s', s)
#        cv2.imshow('lev_v', v)
#        cv2.waitKey(0)

        _, lev = self.levels(minv, maxv, gamma, lev)
        if _ is not None:
            img_hsv[:,:,level] = lev

        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def _blur(self, img, block=(5,5)):
        return cv2.GaussianBlur(img, block, 0)

    def blur(self, block=(5,5)):
        self._img = self._blur(self._img, block)

    def hsv_threshold(self, img=None):
        if img is None:
            img = self._img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_h = img_hsv[:,:,0]
        hsv_s = img_hsv[:,:,1]
        hsv_v = img_hsv[:,:,2]
#        cv2.imshow('th_img', img)
#        cv2.imshow('th_h', hsv_h)
#        cv2.imshow('th_s', hsv_s)
#        cv2.imshow('th_v', hsv_v)
        ret,otsu = cv2.threshold(hsv_s,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu

    def threshold(self, gray=None):
        if gray is None:
            gray = self.gray()
        ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu

    i = 0

    @staticmethod
    def whiteness(roi):
        r = ((np.array([0,0,0]), np.array([80,80,80])))
        black = cv2.inRange(roi, r[0], r[1])
#        cv2.imshow('im{}'.format(Preprocessor.i), roi)
        Preprocessor.i += 1
        black_inv = cv2.bitwise_not(black)

        white = np.zeros((len(roi), len(roi[0]), 3), np.uint8)
        white[:] = (255,255,255)

        roi = cv2.bitwise_and(roi, roi, mask=black_inv)
        white_roi = cv2.bitwise_and(white, white, mask=black)
        return cv2.add(roi, white_roi)

    def gray(self, img=None):
        if img is None:
            img = self._img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def bgr(self, img=None):
        if img is None:
            img = self._img
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def hsv(self, img=None):
        if img is None:
            img = self._img
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def isblack(self, border=10, limit=150.0):
        if self._isblack is None:
            # TODO: Caclulate 10%
            # cv2.mean(img, mask=mask)
            mean_top = np.average(cv2.mean(self._img[:border])) < limit
            mean_bot = np.average(cv2.mean(self._img[-border:])) < limit
            self._isblack = mean_top and mean_bot
        return self._isblack

    def skew(self, minv, maxv, gamma=1.0):
        #self._img = self.bgr(self.hsv_threshold())
        lev = self.hsv_levels(minv, maxv, gamma)
        gray = self.gray(self._blur(lev))
#        cv2.imshow('skew_lev', lev)
        mask = cv2.inRange(gray, 240, 255)

        # Расширяем границы картинки для лучшего определения границ
        cols, rows, _ = lev.shape
        gray_ext = np.zeros((cols+10, rows+10), np.uint8)
        if not self.isblack():
            gray_ext = 255-gray_ext
        gray_ext[5:-5,5:-5] = gray

        mask_ext = cv2.inRange(gray_ext, 240, 255)

        rect = None
        for m in (gray_ext, 255-mask_ext):
            #n = cv2.moments(m)
            #print(n['mu11']/n['mu02'])
            _,contours,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, 2)
            if not contours:
                continue
            cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            rect = cv2.minAreaRect(cnt)
            if abs(rect[2]) > 45:
                rect = (rect[0], rect[1], 90.0 + rect[2])
            if abs(rect[2]) < 7 and abs(rect[2]) != 0: # != -90.0 and rect[2] != 0.0:
                print('Skew: %.3f°' % rect[2])
                break

        if rect is None or rect[2] == 0:
            return None
        if abs(rect[2]) > 7:
            print('Rect', rect[2])
            return None
        return ((rect[0][0]-5.0, rect[0][1]-5.0), rect[1], rect[2])

    def rotate(self, rect, border=20, fill=(255,255,255)):
        if not rect:
            return

        M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)

#        cv2.cv.WarpAffine(cv2.cv.fromarray(self._img), dst, cv2.cv.fromarray(M), flags=cv2.INTER_LINEAR+8, fillval=fill)
        self._img = cv2.warpAffine(self._img, M, (self._img.shape[1], self._img.shape[0]), flags=cv2.INTER_LINEAR+8)

            #img_lev = pre_levels(img_scale, 200, 255)
#            img_lev = pre_levels(img_scale, 0, 170)

        self._img[:border]    = Preprocessor.whiteness(self._img[:border])
        self._img[-border:]   = Preprocessor.whiteness(self._img[-border:])
        self._img[:,-border:] = Preprocessor.whiteness(self._img[:,-border:])
        self._img[:,:border]  = Preprocessor.whiteness(self._img[:,:border])

    def segmentate(self):
        self._cnts = []

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

class Rect:
    def __init__(self, rect, area, cnt, img):
        self._rect = rect
        self._area = area
        self._cnt = cnt
        self._img = img

    def __repr__(self):
        return '<Rect ({},{}) ({}x{}) #{}>'.format(self.x, self.y, self.w, self.h, self.area)

    @property
    def img(self):
        return self._img

    @property
    def x(self):
        return self._rect[0]

    @property
    def y(self):
        return self._rect[1]

    @property
    def w(self):
        return self._rect[2]

    @property
    def h(self):
        return self._rect[3]

    @property
    def rect(self):
        return self._rect

    @property
    def area(self):
        return self._area

    @property
    def cnt(self):
        return self._cnt

    @property
    def wx(self):
        return self.x + self.w

    @property
    def hy(self):
        return self.y + self.h

    @property
    def top(self):
        return (self.x, self.y)

    @property
    def bot(self):
        return (self.wx, self.hy)

    def __get__(self, obj, klass):
        print(self, obj, klass)
        return super().__get__(obj, klass)


class Group:
    def __init__(self, rect):
        if not isinstance(rect, Rect):
            raise TypeError('Param {} is not instance of Rect'.format(rect))
        self._r = rect.rect
        self._img = None
        self._maxwx = [rect.wx, rect.wx+1, rect.wx+3]
        self._rects = np.array([rect])

    def __repr__(self):
        return '<Group ({},{} {},{}) {}x{} #{}>'.format(self.x, self.y, self.wx, self.hy, self.w, self.h, len(self._rects))

    def __lshift__(self, rect):
        if not isinstance(rect, Rect):
            raise TypeError('Param {} is not instance of Rect'.format(rect))
        self._rects = np.append(self._rects, [rect], 0)
        self._r = [
            min(self._r[0], rect.x),
            min(self._r[1], rect.y),
            max(self.x+self.w, rect.wx) - self.x,
            max(self.y+self.y, rect.hy) - self.y,
        ]
        self._maxwx = [
            max(self._maxwx[0], rect.wx),
            max(self._maxwx[1], rect.wx+1),
            max(self._maxwx[2], rect.wx+3),
        ]

    def __iter__(self):
        self._iter_pos = 0
        self._iter_max = len(self._rects)
        return self

    def __next__(self):
        if self._iter_pos == self._iter_max:
            raise StopIteration
        r = self._rects[self._iter_pos]
        self._iter_pos += 1
        return r

    @property
    def img(self):
        if self._img is not None:
            return self._img
        return self._rects[0].img

    @img.setter
    def img(self, value):
        self._img = value

    @property
    def last(self):
        return self._rects[-1]

    @property
    def x(self):
        return self._r[0]

    @property
    def y(self):
        return self._r[1]

    @property
    def w(self):
        return self._r[2]

    @property
    def h(self):
        return self._r[3]

    @property
    def wx(self):
        return self.x + self.w

    @property
    def hy(self):
        return self.y + self.h

    @property
    def area(self):
        print(self._rects)
        return 0 #return cv2.contourArea(self)

    def ispart(self, nxt):
        last = self.last
        return  nxt.x < self._maxwx[0] or \
               (nxt.x == self._maxwx[1] and last.h/nxt.h > 0.9) or \
               (nxt.x < self._maxwx[2] and nxt.w < nxt.h and (nxt.area < 60 or last.h/nxt.h > 1.5))

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

class OCR:
    def __init__(self, ABC, model):
        self._ABC = ABC
        self._model = model
        self._class_n = len(ABC)

    def load(self, datafile):
        self._model.load(datafile)

    def train(self, dname, datafile, percent=0.98):
        digits = None
        labels = None
        idx = 0
        for l in self._ABC:
            digits0, labels0 = self.load_digits('{}/{}'.format(dname, l), idx)
            if digits is None:
                digits = digits0
                labels = labels0
            else:
                digits = np.append(digits, digits0, axis=0)
                labels = np.append(labels, labels0, axis=0)
            idx += 1

        self._digits = digits

        # shuffle digits
        rand = np.random.RandomState(int(np.random.rand()*100)) #321)
        shuffle = rand.permutation(len(digits))
        digits, labels = digits[shuffle], labels[shuffle]

        digits2 = list(map(self.deskew, digits))
        samples = self.preprocess_hog(digits2)

        if percent == 1.0:
            samples_train = samples
            labels_train = labels
        else:
            train_n = int(percent*len(samples))
            cv2.imshow('test set', mosaic(10, digits[train_n:]))
            digits_train, digits_test = np.split(digits2, [train_n])
            samples_train, samples_test = np.split(samples, [train_n])
            labels_train, labels_test = np.split(labels, [train_n])

        self._model.train(samples_train, labels_train)

        if percent != 1.0:
            resp, vis = self.evaluate(digits_test, samples_test, labels_test)
            print([self._ABC[int(c)] for c in labels_test])
            print([self._ABC[int(c)] for c in resp])
            cv2.imshow('SVM test', vis)

        self._model.save(datafile)
        cv2.imwrite(datafile+'.png', mosaic(25, self._digits))

    @property
    def digits(self):
        return self._digits

    def split2d(img, cell_size, flatten=True):
        h, w = img.shape[:2]
        sx, sy = cell_size
        cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
        cells = np.array(cells)
        if flatten:
            cells = cells.reshape(-1, sy, sx)
        return cells

    def load_digits(self, dname, idx):
        print('loading from "%s"...' % (dname))
        digits = None
        for f in os.listdir(dname):
            digits_img = cv2.imread('{}/{}'.format(dname, f), 0)
            if digits is None:
                digits = np.array([digits_img])
            else:
                digits = np.append(digits, [digits_img], axis=0)
#            digits.append(split2d(digits_img, (SZ, SZ)))
        labels = np.repeat(idx, len(digits))
        return digits, labels

    def deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*self._class_n*skew], [0, 1, 0]])
#        img = cv2.warpAffine(img, M, (self._class_n, self._class_n), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def predict(self, samples):
        return self._model.predict(samples)

    def labels(self, predicted):
        return [self._ABC[int(i)] for i in predicted]

    def evaluate(self, digits, samples, labels):
        resp = self.predict(samples)
        print(labels)
        err = (labels != resp).mean()
        print('error: %.2f %%' % (err*100))

        confusion = np.zeros((self._class_n, self._class_n), np.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print('confusion matrix:')
#        print(confusion)
        print()

        vis = []
        for img, flag in zip(digits, resp == labels):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if not flag:
                img[...,:2] = 0
            vis.append(img)
        return resp, mosaic(10, vis)

    def preprocess_simple(self, digits):
        return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

    def preprocess_hog(self, digits):
        samples = []
        for img in digits:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            bin_n = 16
            bin = np.int32(bin_n*ang/(2*np.pi))
            bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
            mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
#            bin_cells = bin[:26,:26], bin[26:,:26], bin[:26,26:], bin[26:,26:]
#            mag_cells = mag[:26,:26], mag[26:,:26], mag[:26,26:], mag[26:,26:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
            hist = np.hstack(hists)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps

            samples.append(hist)
        return np.float32(samples)

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return it.zip_longest(fillvalue=fillvalue, *args)

def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = imgs.__next__()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

def water(img, thresh):
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers += 1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    return sure_fg, sure_bg, markers, img

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
