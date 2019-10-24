#!/usr/bin/env python

import numpy as np
import scipy.stats
import random
import time
import cv2
import logging
from numba import jit

from utility import setup_logger

@jit
def create_space(center, size, element_count):
	lower = center - size / 2.
	upper  = lower + size
	return np.linspace(lower, upper, element_count)

@jit
def create_complex_space(center, size, element_count):
	return (create_space(center[0], size[0], element_count[0]),
			create_space(center[1], size[1], element_count[1]))
@jit
def mandelbrot_depth(c, max_depth):
	z = c
	for i in range(max_depth):
		if z.real * z.real + z.imag * z.imag >= 4.:
			return i
		z = z * z + c
	return 0

@jit
def create_depth_list(center, size, element_count, max_depth):
	space_real, space_imag = create_complex_space (center, size, element_count)
	depth_list = np.empty(element_count[0] * element_count[1], np.uint32)
	for y in range(element_count[1]):
		for x in range(element_count[0]):
			depth_list[x + y * element_count[0]] =	mandelbrot_depth(space_real[x] + 1j * space_imag[y], max_depth)
	return depth_list

@jit
def create_mandelbrot(center, size, element_count, max_depth, loop_depth, start_hue):
	space_real, space_imag = create_complex_space (center, size, element_count)

	depth_list = create_depth_list(center, size, element_count, max_depth)
	depth_nonzero = depth_list[np.nonzero(depth_list)]
	if len(depth_nonzero) > 0:
		min_depth = np.amin(depth_nonzero)
	else:
		min_depth = 0
	max_depth = np.amax(depth_list)

	hsv_img = np.empty((element_count[1], element_count[0], 3), np.uint8)
	for y in range(element_count[1]):
		for x in range(element_count[0]):
			depth =	 depth_list[x + y * element_count[0]]
			hsv_img[y,x] = assign_hsv_color(depth, min_depth, max_depth, start_hue)
	return hsv_img

@jit
def assign_hsv_color(depth, min_depth, max_depth, start_hue):
	if depth == 0:
		hue = 0
		sat = 0
		val = 0
	else:
		factor = (depth - min_depth) / (max_depth - min_depth)
		# factor = 1. - 10**(-depth / max_depth)
		# factor = max(1e-3, min(1., factor))
		hue = (start_hue + int(180. * factor)) % 180
		sat = 255
		val = 255 
	return np.array([hue, sat, val], np.uint8)

log = logging.getLogger()

class Mandelbrot:

	center = (0.41825763942621236, -0.34087020388354944)
	shape_divider = 1e12
	start_hue = 0
	def __init__(self, shape, max_depth):
		self.size = (shape[0] / self.shape_divider, shape[1] / self.shape_divider)
		self.shape = shape
		self.max_depth = max_depth
		self.loop_depth = max_depth / 10

	def randomize_center(self):
		self.center = (random.uniform(-2., 2.), random.uniform(-2., 2.))

	def randomize_start_hue(self):
		self.start_hue = random.randint(0, 180)
	
	def randomize_zoom(self):
		self.shape_divider = random.uniform(1e-5, 1e-14)

	def randomize_depth(self):
		self.max_depth = random.randint(500, 2000)

	def set_depth(self, value):
		self.max_depth = value

	def mod_depth(self, offset):
		self.max_depth += offset

	def zoom(self, factor):
		self.shape_divider *= factor
		self.size = (self.shape[0] / self.shape_divider, self.shape[1] / self.shape_divider)

	def estimate_entropy(self):
		depth_list = create_depth_list(self.center, self.size, (50, 50), self.max_depth)
		_val, counts = np.unique(depth_list, return_counts = True)
		return scipy.stats.entropy(counts)

	def find_poi(self, min_entropy = 4., max_tries = 1000):
		log.info("Searching POI...")
		tries = 0
		while tries < max_tries:
			tries += 1
			self.randomize_center()
			entropy = self.estimate_entropy()
			if entropy >= min_entropy:
				log.info(f"Found POI @ {self.center[0] + 1j * self.center[1]}, tries = {tries}, entropy = {entropy}")
				return True
		return False


	def random_snapshot(self, min_entropy, max_tries):
		while not self.find_poi(min_entropy, max_tries):
			self.randomize_start_hue()
			self.snapshot()

	def snapshot(self):
		ts = time.localtime()
		rgb_img = self.create_rgb()
		return cv2.imwrite(f"{ts.tm_year}-{ts.tm_mon:02}-{ts.tm_mday:02}_{ts.tm_hour:02}-{ts.tm_min:02}-{ts.tm_sec:02}.png", rgb_img)
	
	def create_rgb(self):
		log.info(f"Computing mandelbrot: shape = {self.shape[0]} x {self.shape[1]}, max_depth = {self.max_depth}")
		start = time.time()
		hsv_img = create_mandelbrot(self.center, self.size, self.shape, self.max_depth, self.loop_depth, self.start_hue)
		diff = time.time() - start
		log.info("Computation took {0:.2f}s".format(diff))
		return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
	SHAPE = (1920, 1080)
	MAX_DEPTH = 1500

	setup_logger()

	mb = Mandelbrot(SHAPE, MAX_DEPTH)
	for i in range(1000):
		mb.randomize_zoom()
		mb.randomize_depth()
		mb.random_snapshot()
