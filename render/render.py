import numpy as np
import sys
import random
import pygame

import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 40
SCREENWIDTH  = 400
SCREENHEIGHT = 720

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('rlSpeed')

PIPE_WIDTH = 20
PIPE_NUM = 4
PIP_INTER = 100

PIPGAP = 12
BASEY = 12



MAX_ACC = 3
CONTROL_STEP = 0.1

class Render:

    def __init__(self):
        self.score = 0.1
        self.pips = self.get_random_pips()
        self.pipVx = -25
        pass


    def new_frame(self, speed):
        terminal = False
        reward = 0.1
        # draw

        # calculate reward

        FPSCLOCK.tick(FPS)
        return terminal, reward, image_data
        pass


    def get_random_pips():
        speed_list = [0]
        speed_list = speed_list +  max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)
        speed_list = speed_list +  max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)
        speed_list = speed_list +  max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)
        speed_list = speed_list +  max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)
        pip0 = {'x':PIP_INTER * 0, 'speed':speed_list[0]}
        pip1 = {'x':PIP_INTER * 1, 'speed':speed_list[1]}
        pip2 = {'x':PIP_INTER * 2, 'speed':speed_list[2]}
        pip3 = {'x':PIP_INTER * 3, 'speed':speed_list[3]}
        pip4 = {'x':PIP_INTER * 4, 'speed':speed_list[4]}

        return [pip0, pip1, pip2, pip3, pip4]
        pass

    def get_next_pip():
        self.pips = self.pips[1:]
        new_speed = self.pips[-1]['speed'] + max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)
        new_x = self.pips[-1]['x'] + 10
        self.pips = self.pips + {'x':new_x, 'speed':new_speed}
