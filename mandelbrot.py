#!/usr/bin/env python

import logging
import math
import random
import time

import cv2
import numpy as np
import scipy.stats
from numba import cuda, jit

from utility import setup_logger

MIN_DEPTH = 100
MAX_DEPTH = 100000


def default_filename():
    ts = time.localtime()
    return f"{ts.tm_year}-{ts.tm_mon:02}-{ts.tm_mday:02}_{ts.tm_hour:02}_{ts.tm_min:02}_{ts.tm_sec:02}"


@cuda.jit(device=True)
def mandelbrot_depth(x, y, max_depth):
    c = complex(x, y)
    z = c
    for i in range(max_depth):
        if z.real * z.real + z.imag * z.imag >= 4.0:
            return i
        z = z * z + c
    return 0


@cuda.jit
def fill_depths(origin_x, origin_y, step_size, max_depth, image):
    (x, y) = cuda.grid(2)
    image[x, y] = mandelbrot_depth(
        origin_x + x * step_size, origin_y + y * step_size, max_depth
    )


@cuda.jit
def fill_color(origin_x, origin_y, step_size, max_depth, buckets, image, image_depth):
    (x, y) = cuda.grid(2)
    depth = mandelbrot_depth(
        origin_x + x * step_size, origin_y + y * step_size, max_depth
    )

    if depth != 0:
        cycles = depth // buckets.shape[0]
        if cycles > 2:
            return
        color = buckets[depth % buckets.shape[0]]
        image[x, y, 0] = color[0]
        image[x, y, 1] = color[1]
        image[x, y, 2] = color[2]
        image_depth[x, y] = depth


def create_mandelbrot(center, shape, step_size, reference_depth, buckets):
    max_depth = min(MAX_DEPTH, max(MIN_DEPTH, 10 * reference_depth))

    origin_x = center.real - shape[0] / 2 * step_size
    origin_y = center.imag - shape[1] / 2 * step_size
    image = np.zeros((shape[0], shape[1], 3), np.uint8)
    image_depth = np.zeros((shape[0], shape[1]), np.uint32)

    device_image = cuda.to_device(image)
    device_image_depth = cuda.to_device(image_depth)
    device_buckets = cuda.to_device(buckets)

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.floor(shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.floor(shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    fill_color[blocks_per_grid, threads_per_block](
        origin_x,
        origin_y,
        step_size,
        max_depth,
        device_buckets,
        device_image,
        device_image_depth,
    )

    image = device_image.copy_to_host()
    image_depth = device_image_depth.copy_to_host()

    depths = image_depth.flatten()
    reference_depth = np.median(depths[np.nonzero(depths)])

    return image, reference_depth


def create_hsv_buckets(amount):
    buckets = np.empty((amount, 3), np.uint8)
    acc = [0.0] * 3
    direction = [random.randint(-1, 1) for i in range(3)]
    for i in range(amount):
        acc = [a + random.uniform(0.0, 0.1) for a in acc]
        for j in range(len(direction)):
            if acc[j] >= random.random():
                acc[j] = 0.0
                direction[j] = random.randint(-1, 1)

        if i == 0:
            hue = random.randint(0, 180)
            sat = random.randint(0, 255)
            val = random.randint(0, 255)
        else:
            step = random.randint(1, 4)
            hue = min(180, max(0, int(buckets[i - 1][0]) + direction[0] * step))
            sat = min(200, max(50, int(buckets[i - 1][1]) + direction[1] * step))
            val = min(200, max(50, int(buckets[i - 1][2]) + direction[2] * step))
        buckets[i] = (hue, sat, val)
    return buckets


log = logging.getLogger(__name__)
logging.getLogger(cuda.__name__).setLevel(logging.WARNING)


class Mandelbrot:

    center = 0.41825763942621236 - 0.34087020388354944j
    step_size = 1.0 / 256.0
    reference_depth = 0
    last_img = None
    last_computation_time = 0.0
    buckets = None

    def __init__(self, bucket_count=1000):
        self.randomize_buckets(bucket_count)

    def get_center(self):
        return self.center

    def get_last_computation_time(self):
        return self.last_computation_time

    def get_step_size(self):
        return self.step_size

    def randomize_center(self):
        self.center = random.uniform(-2.0, 2.0) + 1j * random.uniform(-2.0, 2.0)

    def randomize_zoom(self):
        self.step_size = random.uniform(1e-8, 1e-15)

    def randomize_buckets(self, amount=None):
        if amount is None and self.buckets is not None:
            amount = len(self.buckets)
        self.buckets = create_hsv_buckets(amount)

    def set_depth(self, value):
        self.max_depth = value

    def mod_depth(self, offset):
        self.max_depth += offset

    def zoom(self, factor):
        self.step_size /= factor

    def move(self, units):
        self.center = (self.center.real + units[0] * self.step_size) + 1j * (
            self.center.imag + units[1] * self.step_size
        )

    def random_poi(self, shape):
        while True:
            self.randomize_center()
            self.build_rgb(shape)
            v, c = np.unique(self.last_img, return_counts=True)
            e = scipy.stats.entropy(c, base=2)
            log.info("Entropy: %.2f", e)
            if e >= 2.5:
                break

    def zoomed_sequence(self, count, shape, zoom):
        for i in range(count):
            name = f"seq_{i:06}"
            self.snapshot(shape, name)
            self.zoom(zoom)
            yield i

    def snapshot(self, shape, name=None):
        self.build_rgb(shape)
        if name is None:
            name = default_filename() + ".png"
        log.info(f"Creating '{name}'")
        cv2.imwrite(name, self.get_rgb().swapaxes(0, 1))
        return name

    def build_rgb(self, shape):
        log.info(
            f"Computing mandelbrot: center = {self.center}, shape = {shape[0]} x {shape[1]}, step_size = {self.step_size}, reference_depth={self.reference_depth}"
        )
        start = time.time()
        hsv_img, self.reference_depth = create_mandelbrot(
            self.center,
            shape,
            self.step_size,
            self.reference_depth,
            self.buckets,
        )
        self.last_computation_time = time.time() - start
        log.info("Computation took {0:.2f}s".format(self.last_computation_time))
        self.last_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    def get_rgb(self):
        return self.last_img
