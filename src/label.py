# #!/usr/bin/env python3

import os
import sys
from typing import Tuple
import pygame
from pygame.locals import *

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

MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_classified_cropped_images_train/"

class LabelTool:
    def __init__(self, size: Tuple[int,int], flags: int, background_color: Tuple[int,int,int]) -> None:
        self.size = size
        self.flags = flags
        self.background_color = background_color
        pygame.init()
        self.window = pygame.display.set_mode(size=self.size, flags=self.flags)
        self.window.fill(color=BACKGROUND_COLOR)

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == VIDEORESIZE:
                    self.window.fill(color=self.background_color)
                    pygame.display.flip()

def main():
   label_tool = LabelTool(size=(WINDOW_WIDTH, WINDOW_HEIGHT), flags=FLAGS, background_color=BACKGROUND_COLOR)
   label_tool.run()
   return

if __name__ == "__main__":
    main()