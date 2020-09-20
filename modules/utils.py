# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
from hashlib import md5

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

# separation between points
def pt_sep(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# determine if a point is inside a bounding box
def in_box(pt, box):
    (x1, y1), (x2, y2) = box
    inside = (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)
    print(pt, box, inside)
    return inside

# =============================================================================
# IMAGES
# =============================================================================
def imread_rgb(filepath: str):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def imread_rgba(filepath:str):
    return cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), 
                        cv2.COLOR_BGRA2RGBA)

# =============================================================================
# ARRAYS
# =============================================================================
def array_id(array: np.ndarray):
    return md5(array.tobytes()).hexdigest()

# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == '__main__':
    a = (1, 1)
    b = (10, 10)
    print(pt_sep(a, b))
