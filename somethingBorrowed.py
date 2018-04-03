from __future__ import division, print_function, absolute_import
import glob
import re
import os
import numpy as np
# from . import image, config, download
from skimage.transform import rescale
# from .color import match_lightness
from skimage import color
import tempfile
from string import Template


# Could be moved to a more configurable place
MAX_SIDE = 500
MIN_SIDE = 256

def resize_by_factor(img, factor):
    if factor != 1:
        return rescale(img, factor, mode='constant', cval=0)
    else:
        return img

def extract(classifier, grayscale, chs, max_side=500, min_side=256):
    """
    Extract using dense model (faster).

    classifier: Caffe Classifier object
    grayscale: Input image
    """
    # Set scale so that the longest is max_side
    shorter_side = np.min(grayscale.shape[:2])
    longer_side = np.max(grayscale.shape[:2])
    scale = min(min_side / shorter_side, 1)
    if longer_side * scale >= max_side:
        scale = max_side / longer_side

    #HOLE = 4

    #samples = classifier.blobs['centroids'].data.shape[1]
    size = classifier.blobs['data'].data.shape[2]
    #centroids = np.zeros((1, samples, 2), dtype=np.float32)
    full_shape = grayscale.shape[:2]
    #raw_grayscale = grayscale.copy()
    if scale != 1:
        grayscale = resize_by_factor(grayscale, scale)
    raw_shape = grayscale.shape[:2]

    #st0 = (size - raw_shape[0]) // 2
    #en0 = st0 + raw_shape[0]
    #st1 = (size - raw_shape[1]) // 2
    #en1 = st1 + raw_shape[1]

    grayscale = image.center_crop(grayscale, (size, size))
    data = grayscale[np.newaxis, np.newaxis]
    #scaled_size = size // HOLE
    #scaled_shape = tuple([shi // HOLE for shi in raw_shape[:2]])

    ret = classifier.forward(data=data)

    #print('Output shape', ret['prediction_h_full'].shape)

    data = {key: classifier.blobs[key].data for key in classifier.blobs}

    scaleds = [ret['prediction_h_full'][0].transpose(1, 2, 0),
               ret['prediction_c_full'][0].transpose(1, 2, 0)]

    scaled_combined = np.concatenate(scaleds, axis=-1)

    #full_combined = resize_by_factor(scaled_combined, HOLE / scale)
    full_combined = resize_by_factor(scaled_combined, 1 / scale)
    full_combined = image.center_crop(full_combined, full_shape)
    assert full_combined.shape[:2] == full_shape
    combined = full_combined

    res = []
    cur = 0
    for i, ch in enumerate(chs):
        C = scaleds[i].shape[-1]
        res.append(combined[..., cur:cur + C])
        cur += C

    info = dict(input_shape=full_shape,
                scaled_shape=raw_shape,
                padded_shape=grayscale.shape[:2],
                output_shape=ret['prediction_h_full'].shape,
                min_side=min_side,
                max_side=max_side)

    return tuple(res) + (info,)


def calc_rgb_from_hue_chroma(grayscale, img_h, img_c, param=None, color_boost=1.0):
    bins = img_h.shape[-1]
    hist_h = img_h
    hist_c = img_c

    spaced = (np.arange(bins) + 0.5) / bins

    if param is None:
        c_method = 'median'
        h_method = 'expectation-cf'
    else:
        vv = param.split(':')
        c_method = vv[0]
        h_method = vv[1]

    factor = 1.0

    # Hue
    if h_method == 'mode':
        hsv_h = (hist_h.argmax(-1) + 0.5) / bins
    elif h_method == 'expectation':
        tau = 2 * np.pi
        a = hist_h * np.exp(1j * tau * spaced)
        hsv_h = (np.angle(a.sum(-1)) / tau) % 1.0
    elif h_method == 'expectation-cf':  # with chromatic fading
        tau = 2 * np.pi
        a = hist_h * np.exp(1j * tau * spaced)
        cc = abs(a.mean(-1))
        factor = cc.clip(0, 0.03) / 0.03
        hsv_h = (np.angle(a.sum(-1)) / tau) % 1.0
    elif h_method == 'pixelwise':
        cum_h = hist_h.cumsum(-1)
        draws = (np.random.uniform(size=cum_h.shape[:2] + (1,)) > cum_h)
        z_h = (draws.sum(-1) + 0.5) / bins
        hsv_h = z_h
    elif h_method == 'median':
        cum_h = hist_h.cumsum(-1)
        z_h = ((0.5 > cum_h).sum(-1) + 0.5) / bins
        hsv_h = z_h
    elif h_method == 'once':
        cum_h = hist_h.cumsum(-1)
        z_h = ((np.random.uniform() > cum_h).sum(-1) + 0.5) / bins
        hsv_h = z_h

    # Chroma
    if c_method == 'mode':  # mode
        hsv_c = hist_c.argmax(-1) / bins
    elif c_method == 'expectation':  # expectation
        hsv_c = (hist_c * spaced).sum(-1)
    elif c_method == 'pixelwise':
        cum_c = hist_c.cumsum(-1)
        draws = (np.random.uniform(size=cum_c.shape) > cum_c)
        z_c = (draws.sum(-1) + 0.5) / bins
        hsv_c = z_c
    elif c_method == 'once':
        cum_c = hist_c.cumsum(-1)
        z_c = ((np.random.uniform() > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    elif c_method == 'median':
        cum_c = hist_c.cumsum(-1)
        z_c = ((0.5 > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    elif c_method == 'q75':
        cum_c = hist_c.cumsum(-1)
        z_c = ((0.75 > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    else:
        raise ValueError('Unknown chroma method')

    hsv_c *= factor

    hsv_v = grayscale + hsv_c / 2
    hsv_s = 2 * hsv_c / (2 * grayscale + hsv_c)

    hsv_s *= color_boost

    hsv = np.concatenate([hsv_h[..., np.newaxis],
                          hsv_s[..., np.newaxis],
                          hsv_v[..., np.newaxis]], axis=-1)
    rgb = color.hsv2rgb(hsv.clip(0, 1)).clip(0, 1)
    return rgb


def calc_rgb(classifier, grayscale, param=None,
             min_side=256, max_side=500,
             color_boost=1.0, return_info=False):
    img_h, img_c, info = extract(classifier,
                                 grayscale,
                                 ['prediction_h', 'prediction_c'],
                                 min_side=min_side,
                                 max_side=max_side)

    rgb = calc_rgb_from_hue_chroma(grayscale, img_h, img_c, param=param,
            color_boost=color_boost)

    if return_info:
        return rgb, info
    else:
        return rgb


def colorize(raw_img, param=None, classifier=None, color_boost=1.0,
             min_side=None, max_side=None, return_info=False):

    if classifier is None:
        classifier = load_default_classifier()

    if raw_img.ndim == 3:
        grayscale = raw_img[..., :3].mean(-1)
    else:
        grayscale = raw_img

    input_size = classifier.blobs['data'].data.shape[2]

    if min_side is None:
        min_side = input_size // 2

    if max_side is None:
        max_side = input_size - 12

    rgb, info = calc_rgb(classifier, grayscale,
                         param=param,
                         min_side=min_side, max_side=max_side,
                         color_boost=color_boost,
                         return_info=True)

    # Correct the lightness
    rgb = match_lightness(rgb, grayscale)

    if return_info:
        return rgb, info
    else:
        return rgb