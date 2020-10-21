#!/usr/bin/env python
import logging
import threading

import gi

from mandelbrot import Mandelbrot
from utility import setup_logger

gi.require_version("Gtk", "3.0")
from gi.repository import GdkPixbuf, GLib, Gtk


class Window:
    def __init__(self, mandelbrot):
        self.mandelbrot = mandelbrot
        self.draw_shape = (1024, 768)

        self.builder = Gtk.Builder()
        self.builder.add_from_file("window.glade")
        self.builder.connect_signals(self)

    def run(self):
        self.builder.get_object("window1").show_all()
        Gtk.main()

    def update_position_label(self):
        center = self.mandelbrot.get_center()

        self.builder.get_object("label_position").set_text(
            f"{center.real}/{center.imag}"
        )

    def update_computation_time_label(self):
        self.builder.get_object("label_computation_time").set_text(
            f"{self.mandelbrot.get_last_computation_time():.2f} s"
        )

    def update_step_size_label(self):
        self.builder.get_object("label_step_size").set_text(
            f"{self.mandelbrot.get_step_size()}"
        )

    def redraw_image(self, rebuild=True):
        if rebuild:
            self.mandelbrot.build_rgb(self.draw_shape)
        data = list(
            self.mandelbrot.get_rgb()
            .swapaxes(0, 1)
            .reshape(
                -1,
            )
        )

        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            data, GdkPixbuf.Colorspace.RGB, False, 8, 1024, 768, 1024 * 3
        )
        self.builder.get_object("image1").set_from_pixbuf(pixbuf)
        self.update_computation_time_label()

    def start_zoomed_sequence(self):
        pass

    def add_log_entry(self, text):
        self.builder.get_object("textview_log").get_buffer().insert_at_cursor(
            text + "\n"
        )

    def on_window1_destroy(self, *args):
        Gtk.main_quit()

    def on_window1_show(self, *args):
        self.redraw_image()
        self.update_position_label()
        self.update_step_size_label()

    def on_window1_key_press_event(self, *args):
        event = args[1]
        if event.string == "+":
            self.mandelbrot.zoom(1.25)
            self.update_step_size_label()
            self.redraw_image()
        elif event.string == "-":
            self.mandelbrot.zoom(0.75)
            self.update_step_size_label()
            self.redraw_image()

    def on_image1_button_press_event(self, *args):
        event = args[1]
        offset = (event.x - self.draw_shape[0] / 2, event.y - self.draw_shape[1] / 2)
        self.mandelbrot.move(offset)
        self.redraw_image()
        self.update_position_label()

    def on_button_snapshot_clicked(self, *args):
        SHAPE = (1920, 1080)
        filename = self.mandelbrot.snapshot(SHAPE)
        self.add_log_entry(
            f"Created snapshot: {filename}, size = {SHAPE[0]} x {SHAPE[1]}, time = {self.mandelbrot.get_last_computation_time():.2f}s"
        )

    def on_button_recolor_clicked(self, *args):
        self.mandelbrot.randomize_buckets()
        self.redraw_image()

    def on_button_zoomed_sequence_clicked(self, *args):
        self.start_zoomed_sequence()


if __name__ == "__main__":
    INITIAL_DEPTH = 10000
    setup_logger()
    log = logging.getLogger(__name__)

    mb = Mandelbrot(INITIAL_DEPTH)

    window = Window(mb)
    window.run()
