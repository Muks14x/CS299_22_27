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

    c_method = 'median'
    # h_method = 'expectation'
    h_method = 'mode'

    factor = 1.0

    # Hue
    if h_method == 'mode':
        hsv_h = (tf.cast(tf.argmax(hist_h, -1), dtype=tf.float32) + 0.5) / bins
    elif h_method == 'expectation':  # with chromatic fading
        tau = 2 * np.pi
        a = tf.cast(hist_h, dtype=tf.complex64) * (tf.exp(1j * tau * tf.cast(spaced, dtype=tf.complex64)))
        # cc = tf.abs(tf.reduce_mean(a, -1))
        # factor2 = tf.clip_by_value(cc, 0, 0.03) / 0.03
        hsv_h = (angle(tf.reduce_sum(a, -1)) / tau) % 1.0

    if c_method == 'median':
        cum_c = tf.cumsum(hist_c, -1)
        z_c = (tf.reduce_sum(tf.cast(tf.greater(0.5, cum_c), tf.float32), -1) + 0.5) / bins
        hsv_c = z_c

    hsv_c *= factor

    return hsv_h * bins, hsv_c * bins
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
