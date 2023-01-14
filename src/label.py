#!/usr/bin/env python3

import os
import sys
from typing import Tuple
import pygame
from pygame.locals import *
import random

pygame.init()

# Window Configurations
FPS=144
WINDOW_WIDTH=1000
WINDOW_HEIGHT=800
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

# Button (Next & Prev) Position
BTN_NEXT_X=900
BTN_NEXT_Y=700
BTN_NEXT_WIDTH=50
BTN_NEXT_HEIGHT=30
BTN_PREV_X=50
BTN_PREV_Y=700
BTN_PREV_WIDTH=50
BTN_PREV_HEIGHT=30
# Button (Next & Prev) Font
BTN_TXT_COLOR=(50,50,50)
BTN_FONT="Consolas"
BTN_FONT_SIZE=15
BTN_FONT_BOLD=True
BTN_FONT_ITALIC=False
# Button (Next & Prev) Color
BUTTON_COLOR=(242,140,40)
BUTTON_COLOR_HOVER=(255,172,28)
BUTTON_COLOR_PRESSED=(255,95,31)

# Text Class Index Position
TXT_CLASS_INDEX_POS_X=35
TXT_CLASS_INDEX_POS_Y=35
# Text Class Index Color & Font
TXT_CLASS_INDEX_COLOR=(50,50,50)
TXT_CLASS_INDEX_FONT="Consolas"
TXT_CLASS_INDEX_FONT_SIZE=20
TXT_CLASS_INDEX_FONT_BOLD=True
TXT_CLASS_INDEX_FONT_ITALIC=False

# Text Cls Idx Position
TXT_CLS_IDX_POS_X=200
TXT_CLS_IDX_POS_Y=35
# Text Cls Idx Color & Font
TXT_CLS_IDX_COLOR=(255,192,203)
TXT_CLS_IDX_FONT="Consolas"
TXT_CLS_IDX_FONT_SIZE=20
TXT_CLS_IDX_FONT_BOLD=True
TXT_CLS_IDX_FONT_ITALIC=False

# Text Class Position
TXT_CLASS_POS_X=35
TXT_CLASS_POS_Y=75
# Text Class Color & Font
TXT_CLASS_COLOR=(50,50,50)
TXT_CLASS_FONT="Consolas"
TXT_CLASS_FONT_SIZE=20
TXT_CLASS_FONT_BOLD=True
TXT_CLASS_FONT_ITALIC=False

# Text Class Name Position
TXT_CLASS_NAME_POS_X=200
TXT_CLASS_NAME_POS_Y=75
# Text Class Name Color & Font
TXT_CLASS_NAME_COLOR=(255,192,203)
TXT_CLASS_NAME_FONT="Consolas"
TXT_CLASS_NAME_FONT_SIZE=20
TXT_CLASS_NAME_FONT_BOLD=True
TXT_CLASS_NAME_FONT_ITALIC=False

# MTSD Dataset
MTSD_CLASSES=401
MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_classified_cropped_images_train/"
CLASS_DIRS=os.listdir(path=MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR)
CLASS_INDEX=0

# MTSD Dataset Image Position
IMAGE_WIDTH=100
IMAGE_HEIGHT=100
IMAGE_POS_X=35
IMAGE_POS_Y=135
IMAGE_ROW=3
IMAGE_COLUMN=3
IMAGE_GRANULARITY=30

class Text:
    def __init__(self,
                 font: str,
                 font_size: int,
                 bold: bool,
                 italic: bool,
                 txt: str,
                 txt_color: Tuple[int, int, int],
                 txt_pos: Tuple[int, int],
                 ) -> None:
        self.font = pygame.font.SysFont(name=font, size=font_size, bold=bold, italic=italic)
        self.txt_surf = self.font.render(txt, False, txt_color)
        self.txt_pos = txt_pos
        self.txt_rect = self.txt_surf.get_rect(center=txt_pos)

    def txtshow(self, window) -> None:
        window.blit(self.txt_surf, self.txt_pos)
        return

class Button:
    def __init__(self,
                 x: float,
                 y: float,
                 width: float,
                 height: float,
                 flags: int,
                 font: str,
                 font_size: int,
                 bold: bool,
                 italic: bool,
                 btn_txt: str,
                 btn_txt_color: Tuple[int, int, int],
                 onclick_fn,
                 one_press: bool,
                 ) -> None:
        self.btn_surf = pygame.Surface(size=(width, height), flags=flags)
        self.font = pygame.font.SysFont(name=font, size=font_size, bold=bold, italic=italic)
        self.txt_surf = self.font.render(btn_txt, True, btn_txt_color)
        self.btn_rect = pygame.Rect(x, y, width, height)
        self.onclick_fn = onclick_fn
        self.one_press = one_press
        self.pressed = False

    def process(self, window) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self.btn_surf.fill(color=BUTTON_COLOR)
        if self.btn_rect.collidepoint(mouse_pos):
            self.btn_surf.fill(color=BUTTON_COLOR_HOVER)
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.btn_surf.fill(color=BUTTON_COLOR_PRESSED)
                if self.one_press:
                    self.onclick_fn()
                elif not self.pressed:
                    self.onclick_fn()
                    self.pressed = True
            else:
                self.pressed = False
        self.btn_surf.blit(self.txt_surf, [self.btn_rect.width/2 - self.txt_surf.get_rect().width/2,
                                           self.btn_rect.height/2 - self.txt_surf.get_rect().height/2])
        window.blit(self.btn_surf, self.btn_rect)
        return

class LabelTool:
    def __init__(self,
                 size: Tuple[int,int],
                 flags: int,
                 background_color: Tuple[int,int,int],
                 fps: int
                 ) -> None:
        self.size = size
        self.flags = flags
        self.background_color = background_color
        pygame.display.set_caption("Label Tool")
        self.window = pygame.display.set_mode(size=self.size, flags=self.flags)
        button_next = Button(x=BTN_NEXT_X, y=BTN_NEXT_Y, width=BTN_NEXT_WIDTH, height=BTN_NEXT_HEIGHT, flags=0,
                             font=BTN_FONT, font_size=BTN_FONT_SIZE, bold=BTN_FONT_BOLD, italic=BTN_FONT_ITALIC,
                             btn_txt="Next", btn_txt_color=BTN_TXT_COLOR,
                             onclick_fn=self.imshow_next, one_press=False)
        button_prev = Button(x=BTN_PREV_X, y=BTN_PREV_Y, width=BTN_PREV_WIDTH, height=BTN_PREV_HEIGHT, flags=0,
                             font=BTN_FONT, font_size=BTN_FONT_SIZE, bold=BTN_FONT_BOLD, italic=BTN_FONT_ITALIC,
                             btn_txt="Prev", btn_txt_color=BTN_TXT_COLOR,
                             onclick_fn=self.imshow_prev, one_press=False)
        self.buttons = [button_next, button_prev]
        self.txt_class_index = Text(font=TXT_CLASS_INDEX_FONT, font_size=TXT_CLASS_INDEX_FONT_SIZE,
                                    bold=TXT_CLASS_INDEX_FONT_BOLD, italic=TXT_CLASS_INDEX_FONT_ITALIC,
                                    txt="Class Index: ", txt_color=TXT_CLASS_INDEX_COLOR,
                                    txt_pos=(TXT_CLASS_INDEX_POS_X, TXT_CLASS_INDEX_POS_Y))
        self.txt_cls_idx = Text(font=TXT_CLS_IDX_FONT, font_size=TXT_CLS_IDX_FONT_SIZE,
                                bold=TXT_CLS_IDX_FONT_BOLD, italic=TXT_CLS_IDX_FONT_ITALIC,
                                txt=str(CLASS_INDEX), txt_color=TXT_CLS_IDX_COLOR,
                                txt_pos=(TXT_CLS_IDX_POS_X, TXT_CLS_IDX_POS_Y))
        self.txt_class = Text(font=TXT_CLASS_FONT, font_size=TXT_CLASS_FONT_SIZE,
                              bold=TXT_CLASS_FONT_BOLD, italic=TXT_CLASS_FONT_ITALIC,
                              txt="Class: ", txt_color=TXT_CLASS_COLOR,
                              txt_pos=(TXT_CLASS_POS_X, TXT_CLASS_POS_Y))
        self.txt_class_name = Text(font=TXT_CLASS_NAME_FONT, font_size=TXT_CLASS_NAME_FONT_SIZE,
                                   bold=TXT_CLASS_NAME_FONT_BOLD, italic=TXT_CLASS_NAME_FONT_ITALIC,
                                   txt=CLASS_DIRS[CLASS_INDEX], txt_color=TXT_CLASS_NAME_COLOR,
                                   txt_pos=(TXT_CLASS_NAME_POS_X, TXT_CLASS_NAME_POS_Y))
        self.fps = fps
        self.fps_clk = pygame.time.Clock()
        self.window.fill(color=BACKGROUND_COLOR)
        pygame.display.flip()

    def init_window(self):
        self.txt_class_index.txtshow(window=self.window)
        self.txt_class.txtshow(window=self.window)
        self.update_txt()
        self.imshow(rows=IMAGE_ROW, columns=IMAGE_COLUMN)

    def update_txt(self) -> None:
        self.window.fill(color=BACKGROUND_COLOR)
        self.txt_class_name = Text(font=TXT_CLASS_NAME_FONT, font_size=TXT_CLASS_NAME_FONT_SIZE,
                                   bold=TXT_CLASS_NAME_FONT_BOLD, italic=TXT_CLASS_NAME_FONT_ITALIC,
                                   txt=CLASS_DIRS[CLASS_INDEX], txt_color=TXT_CLASS_NAME_COLOR,
                                   txt_pos=(TXT_CLASS_NAME_POS_X, TXT_CLASS_NAME_POS_Y))
        self.txt_cls_idx = Text(font=TXT_CLS_IDX_FONT, font_size=TXT_CLS_IDX_FONT_SIZE,
                                bold=TXT_CLS_IDX_FONT_BOLD, italic=TXT_CLS_IDX_FONT_ITALIC,
                                txt=str(CLASS_INDEX), txt_color=TXT_CLS_IDX_COLOR,
                                txt_pos=(TXT_CLS_IDX_POS_X, TXT_CLS_IDX_POS_Y))
        self.txt_class_name.txtshow(window=self.window)
        self.txt_cls_idx.txtshow(window=self.window)

    def imshow(self, rows: int, columns: int) -> None:
        class_dir = MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR + CLASS_DIRS[CLASS_INDEX] + '/'
        image_keys = os.listdir(path=class_dir)
        if len(image_keys) > rows*columns:
            indices = random.sample(population=range(0, len(image_keys)), k=rows*columns)
        else:
            indices = list(range(0, len(image_keys)))

        images = []
        for i in indices:
            image = pygame.image.load(class_dir + image_keys[i])
            image = pygame.transform.scale(surface=image, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
            images.append(image)

        for i, image in enumerate(images):
            self.window.blit(image, (IMAGE_POS_X+i%columns*(IMAGE_WIDTH+IMAGE_GRANULARITY),
                                     IMAGE_POS_Y+i//columns*(IMAGE_HEIGHT+IMAGE_GRANULARITY)))
        return

    def imshow_prev(self) -> None:
        global CLASS_INDEX
        if CLASS_INDEX > 0:
            CLASS_INDEX -= 1
            self.update_txt()
            self.imshow(rows=IMAGE_ROW, columns=IMAGE_COLUMN)
        return

    def imshow_next(self) -> None:
        global CLASS_INDEX
        if CLASS_INDEX < MTSD_CLASSES-1:
            CLASS_INDEX += 1
            self.update_txt()
            self.imshow(rows=IMAGE_ROW, columns=IMAGE_COLUMN)
        return

    def run(self) -> None:
        self.init_window()
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == VIDEORESIZE:
                    self.window.fill(color=self.background_color)
                    pygame.display.flip()
            self.txt_class_index.txtshow(window=self.window)
            self.txt_class.txtshow(window=self.window)
            for button in self.buttons:
                button.process(window=self.window)
                pygame.display.flip()
            self.fps_clk.tick(self.fps)
        return

def main():
    label_tool = LabelTool(size=(WINDOW_WIDTH, WINDOW_HEIGHT), flags=FLAGS, background_color=BACKGROUND_COLOR, fps=FPS)
    label_tool.run()
    return

if __name__ == "__main__":
    main()