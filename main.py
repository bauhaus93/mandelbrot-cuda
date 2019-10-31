#!/usr/bin/env python
import pygame
import logging


from mandelbrot import Mandelbrot
from utility import setup_logger

if __name__ == "__main__":
	SNAPSHOT_SIZE = (1920, 1080)
	WINDOW_SIZE = (800, 600)
	INITIAL_DEPTH = 1000
	setup_logger()
	log = logging.getLogger()

	mb = Mandelbrot(INITIAL_DEPTH)
	

	pygame.init()
	screen = pygame.display.set_mode(WINDOW_SIZE)

	quit = False
	needsUpdate = True
	active_seq = None
	while not quit:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				quit = True
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					pos = pygame.mouse.get_pos()
					offset = (int(pos[0] - WINDOW_SIZE[0] / 2),
						  int(pos[1] - WINDOW_SIZE[1] / 2))
					mb.move(offset)
					needsUpdate = True
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_F1:
					mb.snapshot(SNAPSHOT_SIZE)
				elif event.key == pygame.K_F2:
					SEQ_COUNT = 100000
					if active_seq is None:
						log.info("Creating zoomed sequence...")
						active_seq = mb.zoomed_sequence(SEQ_COUNT, WINDOW_SIZE, 1.01)
				elif event.key == pygame.K_F3:
					TRIES = 10000
					if mb.find_poi(WINDOW_SIZE, max_tries = TRIES):
						needsUpdate = True
				elif event.key == pygame.K_F4:
					mb.randomize_buckets()
					needsUpdate = True
				elif event.key == pygame.K_e:
					mb.zoom(2.)
					needsUpdate = True
				elif event.key == pygame.K_q:
					mb.zoom(0.5)
					needsUpdate = True
				elif event.key == pygame.K_d:
					mb.mod_depth(100)
					needsUpdate = True
				elif event.key == pygame.K_f:
					mb.mod_depth(-100)
					needsUpdate = True
		if active_seq:
			try:
				progress = next(active_seq)
				log.info(f"Progress: {progress}/{SEQ_COUNT}")
			except StopIteration:
				log.info("Finished sequence")
				active_seq = None
			needsUpdate = True

		if needsUpdate:
			needsUpdate = False
			mb.build_rgb(WINDOW_SIZE)
			pygame.surfarray.blit_array(screen, mb.get_rgb().swapaxes(0, 1))
			pygame.display.flip()
		if not active_seq:
			pygame.time.wait(100)


