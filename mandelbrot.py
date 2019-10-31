#!/usr/bin/env python

import numpy as np
import scipy.stats
import random
import time
import cv2
import logging
from numba import jit

from utility import setup_logger

def default_filename():
	ts = time.localtime()
	return f"{ts.tm_year}-{ts.tm_mon:02}-{ts.tm_mday:02}_{ts.tm_hour:02}_{ts.tm_min:02}_{ts.tm_sec:02}"

@jit
def create_space(center, size, element_count):
	lower = center - size / 2.
	upper  = lower + size
	return np.linspace(lower, upper, element_count)

@jit
def create_complex_space(center, size, element_count):
	return (create_space(center.real, size[0], element_count[0]),
			create_space(center.imag, size[1], element_count[1]))
@jit
def mandelbrot_depth(c, max_depth):
	z = c
	for i in range(max_depth):
		if z.real * z.real + z.imag * z.imag >= 4.:
			return i
		z = z * z + c
	return 0

@jit
def create_depth_list(center, shape, shape_divider, max_depth):
	size = (shape[0] / shape_divider, shape[1] / shape_divider)
	space_real, space_imag = create_complex_space (center, size, shape)
	depth_list = np.empty(shape[0] * shape[1], np.uint32)
	for y in range(shape[1]):
		for x in range(shape[0]):
			depth_list[x + y * shape[0]] =	mandelbrot_depth(space_real[x] + 1j * space_imag[y], max_depth)
	return depth_list

@jit
def create_mandelbrot(center, shape, shape_divider, max_depth, buckets):
	size = (shape[0] / shape_divider, shape[1] / shape_divider)
	space_real, space_imag = create_complex_space (center, size, shape)

	depth_list = create_depth_list(center, shape, shape_divider, max_depth)

	hsv_img = np.empty((shape[1], shape[0], 3), np.uint8)
	for y in range(shape[1]):
		for x in range(shape[0]):
			depth =	 depth_list[x + y * shape[0]]
			hsv_img[y,x] = assign_hsv_color(depth, buckets)
	return hsv_img

@jit
def assign_hsv_color(depth, buckets):
	if depth == 0:
		col = np.zeros(3, np.uint8)
	else:
		col = buckets[depth % buckets.shape[0]]
	return col

def create_hsv_buckets(amount):
	buckets = np.empty((amount, 3), np.uint8)
	acc = [0.] * 3
	direction = [random.randint(-1, 1)] * 3
	for i in range(amount):
		acc = [a  + random.uniform(0., 0.10) for a in acc]
		for j in range(len(direction)):
			if acc[j] >= random.random():
				acc[j] = 0.
				direction[j] = random.randint(-1, 1)

		if i == 0:
			hue = random.randint(0, 180)
			sat = random.randint(0, 255)
			val = random.randint(0, 255)
		else:
			step = random.randint(1, 8)
			hue = min(180, max(0, buckets[i-1][0] + direction[0] * step))
			sat = min(255, max(50, buckets[i-1][1] + direction[1] * step))
			val = min(255, max(50, buckets[i-1][2] + direction[2] * step))
		buckets[i] = (hue, sat, val)
	return buckets

log = logging.getLogger()

class Mandelbrot:

	center = 0.41825763942621236 - 0.34087020388354944j
	shape_divider = 1e12
	last_img = None
	def __init__(self, max_depth):
		self.max_depth = max_depth
		self.randomize_buckets()

	def randomize_center(self):
		self.center = random.uniform(-2., 2.) + 1j * random.uniform(-2., 2.)

	def randomize_buckets(self, amount = None):
		if amount is None:
			amount = self.max_depth
		self.buckets = create_hsv_buckets(amount)
	
	def randomize_zoom(self):
		self.shape_divider = random.uniform(1e5, 1e14)

	def randomize_depth(self):
		self.max_depth = random.randint(100, 2000)

	def set_depth(self, value):
		self.max_depth = value

	def mod_depth(self, offset):
		self.max_depth += offset

	def zoom(self, factor):
		self.shape_divider *= factor
	
	def move(self, units):
		self.center = (self.center.real + units[0] / self.shape_divider) + 1j * (self.center.imag + units[1] / self.shape_divider)

	def estimate_entropy(self, shape, points = 50):
		fact = max(shape[0] / points, shape[1] / points)
		depth_list = create_depth_list(self.center, (50, 50), self.shape_divider * fact, self.max_depth)
		_val, counts = np.unique(depth_list, return_counts = True)
		return scipy.stats.entropy(counts)

	def find_poi(self, shape, min_entropy = 4., max_tries = 1000):
		log.info(f"Searching POI...")
		tries = 0
		old_center = self.center
		max_entropy = 0.
		while tries < max_tries:
			tries += 1
			self.randomize_center()
			entropy = self.estimate_entropy(shape)
			if entropy >= min_entropy:
				log.info(f"Found POI @ {self.center}, tries = {tries}, entropy = {entropy}")
				return True
			elif entropy > max_entropy:
				max_entropy = entropy
		log.info(f"No POI found after {max_tries} tries, max entropy found was {max_entropy}")
		self.center = old_center
		return False

	def zoomed_sequence(self, count, shape, zoom_factor):
		for i in range(count):
			self.snapshot(shape, f"seq_{i:05d}")
			self.zoom(zoom_factor)
			if i % 10 == 0:
				yield i
		yield 

	def random_snapshot(self, shape, min_entropy, max_tries):
		self.randomize_zoom()
		self.randomize_depth()
		if self.find_poi(shape, min_entropy, max_tries):
			self.randomize_buckets()
			self.snapshot(shape)

	def snapshot(self, shape, name = None):
		self.build_rgb(shape)
		if name is None:
			name = default_filename()
		log.info(f"Creating '{name}.png'")
		return cv2.imwrite(name + ".png", self.last_img)
	
	def build_rgb(self, shape):
		log.info(f"Computing mandelbrot: shape = {shape[0]} x {shape[1]}, max_depth = {self.max_depth}")
		log.info(f"Center: {self.center}")
		start = time.time()
		hsv_img = create_mandelbrot(self.center, shape, self.shape_divider, self.max_depth, self.buckets)
		diff = time.time() - start
		log.info("Computation took {0:.2f}s".format(diff))
		self.last_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

	def get_rgb(self):
		return self.last_img


if __name__ == "__main__":
	SHAPE = (1920, 1080)
	MAX_DEPTH = 1500

	setup_logger()
	# random.seed("HELLO")

	mb = Mandelbrot(MAX_DEPTH)
	for i in range(1000):
		mb.random_snapshot(SHAPE, 1., 10000)
