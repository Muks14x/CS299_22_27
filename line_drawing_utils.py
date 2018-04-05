from __future__ import print_function
import numpy as np
import cv2
from glob import glob
import os
import sys


def get_image(image_path, size=256):
    return transform(cv2.imread(image_path, 1), size)


def transform(image, size=256):
    cropped_image = cv2.resize(image, (size, size))
    return np.array(cropped_image)


def get_line_drawing(image):
    return cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, blockSize=9, C=2)


def get_image_dirs():
    li = glob(os.path.join("imgs*", "*.jpg"))
    li.sort()
    return li


def imwriteScaled(name, img, scale=True):
    print("saving img " + name)
    if scale:
        cv2.imwrite(name, img * 255)
    else:
        cv2.imwrite(name, img)


## Convert from BGR to HSV/HCL
def bgr2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# h - 0 to 180, c & l - 0 to 1 (float64)
def hsv2hcl(image):
    h0, s0, v0 = [image[:, :, i] for i in range(3)]
    s = s0.astype(np.float64) / 256.0
    v = v0.astype(np.float64) / 256.0
    c = s * v
    l = v - c / 2
    return np.stack([h0, c, l], axis=2)


def hcl2Hist(image, numBuckets=32):
    h0, c0, l0 = [image[:, :, i] for i in range(3)]

    # now h and c are integers from [0, 32)
    h = (h0 * numBuckets / 180.1).astype(int)
    c = (c0 * numBuckets).astype(int)

    # l is an int from [0-256)
    l = (l0 * 256).astype(int)
    return h, c, l


# inputs: h, c, l like from hcl2Hist
def Hist2hcl(h0, c0, l0, numBuckets=32):
    l = l0 / 256.0

    c = c0.astype(np.float64) / numBuckets
    h = (h0 * 180.1 / numBuckets).astype(np.uint8)

    return np.stack([h, c, l], axis=2)


def hcl2hsv(image):
    h0, c0, l0 = [image[:, :, i] for i in range(3)]
    v = (l0 + c0 / 2)
    # hsv_s = 2 * hsv_c / (2 * grayscale + hsv_c)
    s = 2 * c0 / (2 * l0 + c0)
    # s = (c0 / v)
    v *= 256
    s *= 256
    return np.stack([h0, s, v], axis=2).astype(np.uint8)


def hsv2bgr(image):
    image = np.squeeze(image)
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


# Bad writing style:
# functions don't verify input format, so double check changes to code

# Returns h, c, l
# h & c - ints from [0, 32)
# l - int from [0, 256)
def bgr2Hist(image):
    return hcl2Hist(hsv2hcl(bgr2hsv(image)))


def Hist2bgr(h, c, l, upScaleL=False):
    if upScaleL:
        l = l * 256.0
    p = Hist2hcl(h, c, l)
    q = hcl2hsv(p)
    return hsv2bgr(q)


# class image:
# def __init__(self, bgr):
#     if len(bgr.shape) != 3:
#         raise ValueError("muks: Rank of input to the image() class isn\'t 3. input shape: " + str(bgr.shape))
#
#     self.B, self.G, self.R = [bgr[:, :, i] for i in range(3)]
#     hsv = bgr2hsv(bgr)
#     self.H, self.S, self.V = [hsv[:, :, i] for i in range(3)]

## Convert HCL to sparse matrix (histogram)
## Convert from (predicted) histogram to HCL
## Convert the HCL to BGR

# def bgr2lab(image):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = np.split(lab, 3, 2)
#     return l, a, b

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    data = get_image_dirs()
    out_size = 256

    if len(sys.argv) > 1:
        if sys.argv[1] == "-o":
            out_size = int(sys.argv[2])

    if not os.path.exists("out_imgs"):
        os.makedirs("out_imgs")

    images = np.array([get_image(sample_file, out_size) for sample_file in data])
    line_drawings = np.array([get_line_drawing(img) for img in images])
    line_drawings = np.expand_dims(line_drawings, 3)
    output = np.tile(line_drawings, 3)

    for i in range(len(data)):
        cv2.imwrite('out_' + data[i], output[i])
