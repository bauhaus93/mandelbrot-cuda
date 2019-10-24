#!/usr/bin/env python
import pygame
import logging


from mandelbrot import Mandelbrot
from utility import setup_logger

if __name__ == "__main__":
	# WINDOW_SIZE = (1920, 1080)
	WINDOW_SIZE = (800, 600)
	INITIAL_DEPTH = 1000
	setup_logger()

	mb = Mandelbrot(WINDOW_SIZE, INITIAL_DEPTH)
	

	pygame.init()
	screen = pygame.display.set_mode(WINDOW_SIZE)

	quit = False
	needsUpdate = True
	while not quit:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				quit = True
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_e:
					mb.zoom(2.)
					needsUpdate = True
				elif event.key == pygame.K_r:
					mb.zoom(0.5)
					needsUpdate = True
				elif event.key == pygame.K_d:
					mb.mod_depth(100)
					needsUpdate = True
				elif event.key == pygame.K_f:
					mb.mod_depth(-100)
					needsUpdate = True
				elif event.key == pygame.K_t:
					mb.randomize_start_hue()
					needsUpdate = True
				elif event.key == pygame.K_g:
					if mb.find_poi(4., 10000):
						needsUpdate = True

		if needsUpdate:
			needsUpdate = False
			mb_img = mb.create_rgb().swapaxes(0, 1)
			pygame.surfarray.blit_array(screen, mb_img)
			pygame.display.flip()

		pygame.time.wait(100)


