#!/usr/bin/env python

import numpy as np
import time
import cv2
from numba import jit

@jit
def mandelbrot_depth(c, max_depth):
    z = c
    for i in range(max_depth):
        if z.real * z.real + z.imag * z.imag >= 4.:
            return i
        z = z * z + c
    return 0

@jit
def check_mandelbrot(center, pixel_shape, shape, max_depth):
    min_x = center[0] - shape[0] / 2.
    max_x  = min_x + shape[0]
    space_real = np.linspace(min_x, max_x, pixel_shape[0])

    min_y = center[1] - shape[1] / 2.
    max_y = min_y + shape[1]
    space_imag = np.linspace(min_y, max_y, pixel_shape[1])

    output = np.empty(pixel_shape)
    for i in range(pixel_shape[0]):
        for j in range(pixel_shape[1]):
            output[i,j] = mandelbrot_depth(space_real[i] + 1j * space_imag[j], max_depth)
    return output

class Mandelbrot:

    center = (0.41825763942621236, -0.34087020388354944)
    shape_divider = 1e14
    def __init__(self, shape, max_depth):
        self.size = (shape[0] / self.shape_divider, shape[1] / self.shape_divider)
        self.shape = shape
        self.max_depth = max_depth

    def snapshot(self):
        depth = self.compute()
        return self.depth_to_pixels(depth)

    def compute(self):
        print(f"Computing mandelbrot: shape = {self.shape[0]} x {self.shape[1]}, max_depth = {self.max_depth}")
        start = time.time()
        depth_list = check_mandelbrot(self.center, self.shape, self.size, self.max_depth)
        diff = time.time() - start
        print(f"Computation took {diff}s")
        return depth_list

    def depth_to_pixels(self, depth_array):
        output = np.empty((depth_array.shape[0], depth_array.shape[1], 3), np.uint8)
        print(output.shape)
        for i in range(depth_array.shape[1]):
            line = np.array([assign_color(e) for e in depth_array[i,:]])
            output[i,:] = line
        return output

def assign_color(depth):
    return np.array([depth % 256, depth % 256, depth % 256], np.uint8)
 

SHAPE = (1920, 1080)
MAX_DEPTH = 400

mb = Mandelbrot(SHAPE, MAX_DEPTH)
snap = mb.snapshot()

cv2.imwrite("output.png", snap)

cv2.imshow('Test image', snap)
cv2.waitKey(0)
cv2.destroyAllWindows()



