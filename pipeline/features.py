"""
Feature extraction module.
Handles stem detection, extreme point detection, and line drawing.
"""
import numpy as np
import cv2
from skimage.measure import label, regionprops


def get_stems(A_thin, blobs):
    """
    Group-1 stem detection (Kumar-style):
    - Identify loop skeleton pixels adjacent to blob (hole)
    - Remove those loop pixels
    - Remaining connected components are stems/tails
    """
    A_thin = A_thin.astype(np.uint8)
    blobs = blobs.astype(np.uint8)

    # 1) Find skeleton pixels adjacent to the blob (these belong to the loop)
    kernel = np.ones((3, 3), np.uint8)
    blobs_dil = cv2.dilate(blobs, kernel, iterations=1)
    loop_pixels = (A_thin & blobs_dil).astype(np.uint8)

    # 2) Remove loop pixels -> candidate stems
    stems_candidate = (A_thin & (1 - loop_pixels)).astype(np.uint8)

    # 3) Label remaining components as stems (filter tiny junk)
    labels = label(stems_candidate, connectivity=2)
    props = regionprops(labels)

    stems_img = np.zeros_like(A_thin, dtype=np.uint8)
    centroids = []

    MIN_STEM_PIXELS = 8  # tune 6–12 if needed

    for r in props:
        if r.area < MIN_STEM_PIXELS:
            continue
        coords = r.coords
        stems_img[coords[:, 0], coords[:, 1]] = 1
        centroids.append(r.centroid)

    n_stems = len(centroids)
    return stems_img, n_stems, centroids

import numpy as np
from skimage.measure import label, regionprops


import numpy as np
from skimage.measure import label, regionprops

def get_banded_points(A, split=0.5):
    """
    Return banded extreme points on the largest connected component.

    split=0.5 means split at mid-height of the digit's bounding box.
    Returns: TL, BL, TR, BR as (x,y) tuples (x=col, y=row)
    """
    A = (A > 0).astype(np.uint8)

    lab = label(A, connectivity=2)
    props = regionprops(lab)
    if not props:
        return None

    main = max(props, key=lambda r: r.area)
    coords = main.coords  # (y,x)

    ys = coords[:, 0]
    xs = coords[:, 1]

    y_min = int(ys.min())
    y_max = int(ys.max())
    y_mid = int(y_min + split * (y_max - y_min))

    top = coords[coords[:, 0] <= y_mid]
    bot = coords[coords[:, 0] >  y_mid]

    # fallback if one half is empty (can happen on weird crops)
    if len(top) == 0:
        top = coords
    if len(bot) == 0:
        bot = coords

    # TL: leftmost x in top band
    top_xs = top[:, 1]
    tl_x = int(top_xs.min())
    tl_y = int(top[top[:, 1] == tl_x][:, 0].min())

    # TR: rightmost x in top band
    tr_x = int(top_xs.max())
    tr_y = int(top[top[:, 1] == tr_x][:, 0].min())

    # BL: leftmost x in bottom band
    bot_xs = bot[:, 1]
    bl_x = int(bot_xs.min())
    bl_y = int(bot[bot[:, 1] == bl_x][:, 0].max())

    # BR: rightmost x in bottom band
    br_x = int(bot_xs.max())
    br_y = int(bot[bot[:, 1] == br_x][:, 0].max())

    TL = (tl_x, tl_y)
    TR = (tr_x, tr_y)
    BL = (bl_x, bl_y)
    BR = (br_x, br_y)

    return TL, BL, TR, BR


def get_extreme_points(A):
    """
    Compute extremes on the largest connected foreground component.
    Returns None if no foreground exists.
    """
    A = (A > 0).astype(np.uint8)

    # no foreground at all
    if np.sum(A) == 0:
        return None

    lab = label(A, connectivity=2)
    props = regionprops(lab)

    # choose largest connected component
    main = max(props, key=lambda r: r.area)
    coords = main.coords  # (row=y, col=x)

    ys = coords[:, 0]
    xs = coords[:, 1]

    xL = int(xs.min())
    xR = int(xs.max())

    yT_l = int(ys[xs == xL].min())
    yB_l = int(ys[xs == xL].max())
    yT_r = int(ys[xs == xR].min())
    yB_r = int(ys[xs == xR].max())

    return (xL, yT_l), (xL, yB_l), (xR, yT_r), (xR, yB_r)


def draw_line(A, p0, p1):
    """
    Draw a 1px white line between p0=(x0,y0) and p1=(x1,y1) on a copy of A.
    Returns new 0/1 image.
    """
    out = (A.copy() * 255).astype(np.uint8)  # to 0/255
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    cv2.line(out, (x0, y0), (x1, y1), 255, 1)  # draw white line
    return (out == 255).astype(np.uint8)
