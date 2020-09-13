# -*- coding: utf-8 -*-

import time
# from shapely.geometry import Polygon
import cv2
import numpy as np

# =============================================================================
# DECORATORS
# =============================================================================
def timeit(func, *args, **kwargs):
    def wrap(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print(f'\n({func.__name__} finished in {time.time()-t0:.4f} seconds)\n')
        return result
    return wrap

# =============================================================================
# CALCULATIONS
# =============================================================================
def round_base_n(x, base: int = 5):
    return base * round(x/base)

# def calc_iou(box_1, box_2):
#     poly_1 = Polygon(box_1)
#     poly_2 = Polygon(box_2)
#     iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
#     return iou

# separation between points
def pt_sep(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# determine if a point is inside a bounding box
def in_box(pt, box, scale: float = 1.0):
    x1, y1, x2, y2 = (i * scale for i in box)
    inside = (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)
    # if inside:
    #     print('Point:', pt)
    #     print('Box:', (x1, x2, y1, y2))
    #     print('Inside:', inside)
    return inside

# =============================================================================
# IMAGES
# =============================================================================
def imread_rgb(filepath: str):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == '__main__':
    a = (1, 1)
    b = (10, 10)
    print(pt_sep(a, b))