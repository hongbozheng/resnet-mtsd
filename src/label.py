#!/usr/bin/env python3

import os
import sys
from typing import Tuple, List
import pygame
from pygame.locals import *

pygame.init()

# Window Configurations
FPS=144
WINDOW_WIDTH=500
WINDOW_HEIGHT=500
"""
pygame.FULLSCREEN    create a fullscreen display
pygame.DOUBLEBUF     only applicable with OPENGL
pygame.HWSURFACE     (obsolete in pygame 2) hardware accelerated, only in FULLSCREEN
pygame.OPENGL        create an OpenGL-renderable display
pygame.RESIZABLE     display window should be sizeable
pygame.NOFRAME       display window will have no border or controls
pygame.SCALED        resolution depends on desktop size and scale graphics
pygame.SHOWN         window is opened in visible mode (default)
pygame.HIDDEN        window is opened in hidden mode
"""
FLAGS=RESIZABLE
BACKGROUND_COLOR=(192,192,192)

# Button Color Configurations
BUTTON_COLOR=(255,165,0)
BUTTON_COLOR_HOVER=(245, 203, 167)
BUTTON_COLOR_PRESSED=(211, 84, 0)

# Button Font Configurations
BTN_TXT_COLOR=(20,20,20)
BTN_FONT="Consolas"
BTN_FONT_SIZE=15
BTN_FONT_BOLD=False
BTN_FONT_ITALIC=False

MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_classified_cropped_images_train/"
INDEX=0

class Button:
    def __init__(self,
                 x: float,
                 y: float,
                 width: float,
                 height: float,
                 flags: int,
                 btn_txt : str,
                 btn_txt_color: Tuple[int, int, int],
                 font: str,
                 font_size: int,
                 bold: bool,
                 italic: bool,
                 onclick_fn,
                 one_press: bool,
                 fps):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.btn_surface = pygame.Surface(size=(self.width, self.height), flags=flags)
        self.font = pygame.font.SysFont(name=font, size=font_size, bold=bold, italic=italic)
        self.btn_txt = self.font.render(btn_txt, True, btn_txt_color)
        self.btn_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.onclick_fn = onclick_fn
        self.one_press = one_press
        self.pressed = False

    def process(self, window):
        mouse_pos = pygame.mouse.get_pos()
        self.btn_surface.fill(color=BUTTON_COLOR)
        if self.btn_rect.collidepoint(mouse_pos):
            self.btn_surface.fill(color=BUTTON_COLOR_HOVER)
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.btn_surface.fill(color=BUTTON_COLOR_PRESSED)
                if self.one_press:
                    self.onclick_fn()
                elif not self.pressed:
                    self.onclick_fn()
                    self.pressed = True
            else:
                self.pressed = False
        self.btn_surface.blit(self.btn_txt, [self.btn_rect.width/2 - self.btn_txt.get_rect().width/2,
                                             self.btn_rect.height/2 - self.btn_txt.get_rect().height/2])
        window.blit(self.btn_surface, self.btn_rect)

def test():
    print("button pressed")

def test1():
    print("multi-pressed")

class LabelTool:
    def __init__(self,
                 size: Tuple[int,int],
                 flags: int,
                 background_color: Tuple[int,int,int],
                 buttons: List[Button],
                 fps: int) -> None:
        self.size = size
        self.flags = flags
        self.background_color = background_color
        self.buttons = buttons
        self.window = pygame.display.set_mode(size=self.size, flags=self.flags)
        self.window.fill(color=BACKGROUND_COLOR)
        self.fps = fps
        self.fps_clk = pygame.time.Clock()
        pygame.display.flip()

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == VIDEORESIZE:
                    self.window.fill(color=self.background_color)
                    pygame.display.flip()
            for button in self.buttons:
                button.process(window=self.window)
                pygame.display.flip()
            self.fps_clk.tick(self.fps)

def main():
    button_next = Button(x=30, y=30, width=100, height=100, flags=0,
                         btn_txt="Next", btn_txt_color=BTN_TXT_COLOR,
                         font=BTN_FONT, font_size=BTN_FONT_SIZE, bold=BTN_FONT_BOLD, italic=BTN_FONT_ITALIC,
                         onclick_fn=test, one_press=False, fps=FPS)
    button_next1 = Button(x=30, y=140, width=100, height=100, flags=0,
                          btn_txt="Next1", btn_txt_color=BTN_TXT_COLOR,
                          font=BTN_FONT, font_size=BTN_FONT_SIZE, bold=BTN_FONT_BOLD, italic=BTN_FONT_ITALIC,
                          onclick_fn=test1, one_press=True, fps=FPS)
    buttons = [button_next, button_next1]
    label_tool = LabelTool(size=(WINDOW_WIDTH, WINDOW_HEIGHT), flags=FLAGS, background_color=BACKGROUND_COLOR,
                           buttons=buttons, fps=FPS)
    label_tool.run()
    return

if __name__ == "__main__":
    main()