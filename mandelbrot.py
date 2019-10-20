#!/usr/bin/env python

import numpy as np
import time
import cv2
import logging
from numba import jit

from utility import setup_logger

@jit
def mandelbrot_depth(c, max_depth):
    z = c
    for i in range(max_depth):
        if z.real * z.real + z.imag * z.imag >= 4.:
            return i
        z = z * z + c
    return 0

@jit
def create_mandelbrot(center, pixel_shape, shape, max_depth, loop_depth):
    min_x = center[0] - shape[0] / 2.
    max_x  = min_x + shape[0]
    space_real = np.linspace(min_x, max_x, pixel_shape[0])

    min_y = center[1] - shape[1] / 2.
    max_y = min_y + shape[1]
    space_imag = np.linspace(min_y, max_y, pixel_shape[1])

    hsv_img = np.empty((pixel_shape[1], pixel_shape[0], 3), np.uint8)
    for y in range(pixel_shape[1]):
        for x in range(pixel_shape[0]):
            depth =  mandelbrot_depth(space_real[x] + 1j * space_imag[y], max_depth)
            hsv_img[y,x] = assign_hsv_color(depth, loop_depth)
    return hsv_img

@jit
def assign_hsv_color(depth, loop_depth):
    if depth == 0:
        hue = 0
        sat = 0
        val = 0
    else:
        hue = int((255 * (depth % loop_depth)) / depth)
        sat = 255
        val = 255
    return np.array([hue, sat, val], np.uint8)

log = logging.getLogger()

class Mandelbrot:

    center = (0.41825763942621236, -0.34087020388354944)
    shape_divider = 1e14
    loop_depth = 800
    def __init__(self, shape, max_depth):
        self.size = (shape[0] / self.shape_divider, shape[1] / self.shape_divider)
        self.shape = shape
        self.max_depth = max_depth

    def set_depth(self, value):
        self.max_depth = value

    def mod_depth(self, offset):
        self.max_depth += offset

    def zoom(self, factor):
        self.shape_divider *= factor
        self.size = (self.shape[0] / self.shape_divider, self.shape[1] / self.shape_divider)

    def snapshot(self):
        log.info(f"Computing mandelbrot: shape = {self.shape[0]} x {self.shape[1]}, max_depth = {self.max_depth}")
        start = time.time()
        hsv_img = create_mandelbrot(self.center, self.shape, self.size, self.max_depth, self.loop_depth)
        diff = time.time() - start
        log.info("Computation took {0:.2f}s".format(diff))
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    SHAPE = (1920, 1080)
    # SHAPE = (800, 600)
    MAX_DEPTH = 8000

    setup_logger()

    mb = Mandelbrot(SHAPE, MAX_DEPTH)
    snap = mb.snapshot()
    # snap = cv2.GaussianBlur(snap, (3, 3), cv2.BORDER_DEFAULT)
    snap = cv2.fastNlMeansDenoisingColored(snap, h = 25)
    cv2.imwrite("output.png", snap)

    # cv2.imshow('TestImage', snap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



