from __future__ import division, print_function, absolute_import
import numpy as np
from line_drawing_utils import hsv2bgr
import tensorflow as tf

def angle(z):
    if z.dtype == tf.complex128:
        dtype = tf.float64
    else:
        dtype = tf.float32
    x = tf.real(z)
    y = tf.imag(z)
    xneg = tf.cast(x < 0.0, dtype)
    yneg = tf.cast(y < 0.0, dtype)
    ypos = tf.cast(y >= 0.0, dtype)

    offset = xneg * (ypos - yneg) * np.pi

    return tf.atan(y / x) + offset


# def calc_bgr_from_hue_chroma(grayscale, img_h, img_c, param=None, color_boost=1.0):
def get_hc_idx(img_h, img_c, param=None):
    bins = int(img_h.shape[-1])
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
        a = hist_h * tf.exp(1j * tau * tf.cast(spaced, dtype=tf.complex128))
        hsv_h = (np.angle(a.sum(-1)) / tau) % 1.0
    elif h_method == 'expectation-cf':  # with chromatic fading
        tau = 2 * np.pi
        a = hist_h * np.exp(1j * tau * spaced)
        cc = tf.abs(tf.reduce_mean(a, -1))
        factor = tf.clip_by_value(cc, 0, 0.03) / 0.03
        hsv_h = (angle(tf.reduce_sum(a, -1)) / tau) % 1.0
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
    else:
        raise ValueError('Unknown hue method')

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
        cum_c = tf.cumsum(hist_c, -1)
        z_c = (tf.reduce_sum(tf.cast(tf.greater(0.5, cum_c), tf.float32), -1) + 0.5) / bins
        hsv_c = z_c
    elif c_method == 'q75':
        cum_c = hist_c.cumsum(-1)
        z_c = ((0.75 > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    else:
        raise ValueError('Unknown chroma method')

    hsv_c *= factor

    return hsv_h, hsv_c
    # hsv_v = tf.squeeze(grayscale) + hsv_c / 2
    # hsv_s = 2 * hsv_c / (2 * grayscale + hsv_c)

    # hsv_s *= color_boost
    #
    # hsv = np.concatenate([(hsv_h[..., np.newaxis] * 160.0).clip(0, 160),
    #                       (hsv_s[..., np.newaxis] * 256.0).clip(0, 256),
    #                       (hsv_v[..., np.newaxis] * 256.0).clip(0, 256)], axis=-1)
    #
    # bgr = hsv2bgr(hsv)
    #
    # return bgr
