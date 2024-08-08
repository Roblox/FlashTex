import numpy as np

def rgb_to_hls(rgb_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being RGB colours.
    Returns an array of same size, each row being HLS colours.
    Like `colorsys` python module, all values are between 0 and 1.
    NOTE: like `colorsys`, this uses HLS rather than the more usual HSL
    """
    assert rgb_array.ndim == 2
    assert rgb_array.shape[1] == 3
    assert np.max(rgb_array) <= 1
    assert np.min(rgb_array) >= 0

    r, g, b = rgb_array.T.reshape((3, -1, 1))
    maxc = np.max(rgb_array, axis=1).reshape((-1, 1))
    minc = np.min(rgb_array, axis=1).reshape((-1, 1))

    sumc = (maxc+minc)
    rangec = (maxc-minc)

    with np.errstate(divide='ignore', invalid='ignore'):
        rgb_c = (maxc - rgb_array) / rangec
    rc, gc, bc = rgb_c.T.reshape((3, -1, 1))

    h = (np.where(minc == maxc, 0, np.where(r == maxc, bc - gc, np.where(g == maxc, 2.0+rc-bc, 4.0+gc-rc)))
         / 6) % 1
    l = sumc/2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(minc == maxc, 0,
                     np.where(l < 0.5, rangec / sumc, rangec / (2.0-sumc)))

    return np.concatenate((h, l, s), axis=1)


def hls_to_rgb(hls_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being HLS colours.
    Returns an array of same size, each row being RGB colours.
    Like `colorsys` python module, all values are between 0 and 1.
    NOTE: like `colorsys`, this uses HLS rather than the more usual HSL
    """
    ONE_THIRD = 1 / 3
    TWO_THIRD = 2 / 3
    ONE_SIXTH = 1 / 6

    def _v(m1, m2, h):
        h = h % 1.0
        return np.where(h < ONE_SIXTH, m1 + (m2 - m1) * h * 6,
                        np.where(h < .5, m2,
                                 np.where(h < TWO_THIRD, m1 + (m2 - m1) * (TWO_THIRD - h) * 6,
                                          m1)))


    assert hls_array.ndim == 2
    assert hls_array.shape[1] == 3
    assert np.max(hls_array) <= 1
    assert np.min(hls_array) >= 0

    h, l, s = hls_array.T.reshape((3, -1, 1))
    m2 = np.where(l < 0.5, l * (1 + s), l + s - (l * s))
    m1 = 2 * l - m2

    r = np.where(s == 0, l, _v(m1, m2, h + ONE_THIRD))
    g = np.where(s == 0, l, _v(m1, m2, h))
    b = np.where(s == 0, l, _v(m1, m2, h - ONE_THIRD))

    return np.concatenate((r, g, b), axis=1)


def hsv_to_rgb(hsv_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being HSV colours.
    Returns an array of same size, each row being RGB colours.
    Like `colorsys` python module, all values are between 0 and 1.
    """
    assert hsv_array.ndim == 2
    assert hsv_array.shape[1] == 3
    assert np.max(hsv_array) <= 1
    assert np.min(hsv_array) >= 0

    h, s, v = hsv_array.T.reshape((3, -1, 1))
    h = h % 1
    s = s.clip(0, 1)
    v = v.clip(0, 1)
    i = (h * 6).astype("uint8")
    f = (h * 6) - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    # i = i%6
    wh = np.where

    return wh(
        i == 0, np.concatenate((v, t, p), axis=1),
        wh(i == 1, np.concatenate((q, v, p), axis=1),
           wh(i == 2, np.concatenate((p, v, t), axis=1),
              wh(i == 3, np.concatenate((p, q, v), axis=1),
                 wh(i == 4, np.concatenate((t, p, v), axis=1),
                    wh(i == 5, np.concatenate((v, p, q), axis=1),
                       np.full(hsv_array.shape, np.NaN)))))))


def rgb_to_hsv(rgb_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being RGB colours.
    Returns an array of same size, each row being HSV colours.
    Like `colorsys` python module, all values are between 0 and 1.
    """
    assert rgb_array.ndim == 2
    assert rgb_array.shape[1] == 3
    assert np.max(rgb_array) <= 1
    assert np.min(rgb_array) >= 0

    r, g, b = rgb_array.T.reshape((3, -1, 1))
    maxc = np.max(rgb_array, axis=1).reshape((-1, 1))
    minc = np.min(rgb_array, axis=1).reshape((-1, 1))
    v = maxc

    sumc = (maxc+minc)
    rangec = (maxc-minc)

    with np.errstate(divide='ignore', invalid='ignore'):
        rgb_c = (maxc - rgb_array) / rangec
    rc, gc, bc = rgb_c.T.reshape((3, -1, 1))

    h = (np.where(minc == maxc, 0,
                  np.where(r == maxc, bc - gc,
                           np.where(g == maxc, 2.0+rc-bc,
                                    4.0+gc-rc)))
         / 6) % 1
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(minc == maxc, 0, rangec / maxc)

    return np.concatenate((h, s, v), axis=1)