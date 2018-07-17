import numpy as np
import sys
import random
import pygame

import math
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 20
SCREENWIDTH  = 400
SCREENHEIGHT = 900

FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('rlSpeed')

PIPE_WIDTH = 20
PIPE_NUM = 4
PIP_INTER = 100

PIP_GAP = 24
BASEY = 30
PIXELS_SPEED = 90

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

MAX_ACC = 3
CONTROL_STEP = 0.1   # control time step is 0.1s

class Render:

    def __init__(self):
        self.score = 0.1
        self.pips = self.get_random_pips()
        self.pipVx = -25
        self.player = pygame.image.load('render/player.png').convert_alpha()
        self.player_height = self.player.get_height()
        self.fps = 0
        pygame.init()
        pass

    def init(self):
        self.pips = self.get_random_pips()
        self.fps = 0

    def new_frame(self, speed):
        terminal = False
        reward = 0.1
      
        # draw
        SCREEN.fill(BLACK)

        # draw current speed bird
        player_x = PIP_INTER
        player_y = int(speed * PIXELS_SPEED + BASEY - self.player_height/2)
        SCREEN.blit(self.player, (player_x, player_y))

        # calc reward using current speed
        speed_set = self.pips[1]['speed']
        reward = math.exp(abs(speed_set - speed)) / 5
        if abs(speed_set - speed) >= 0.2 and self.fps % 4 == 0:
            terminal = True
            reward = -10

        # calculate reward
        for i,pip in enumerate(self.pips):
            v = pip['speed']
            x = pip['x']
            lower = int(BASEY + PIXELS_SPEED * v - PIP_GAP)
            upper = int(BASEY + PIXELS_SPEED * v + PIP_GAP)
            left = int(x - PIPE_WIDTH/2)
            right = int(x + PIPE_WIDTH/2)
            rect = pygame.Rect((left,0),(PIPE_WIDTH, lower))
            pygame.draw.rect(SCREEN, GREEN, rect)
            rect = pygame.Rect((left,upper),(PIPE_WIDTH, SCREENHEIGHT - upper))
            pygame.draw.rect(SCREEN, GREEN, rect)
            # pipx -= pipVx
            self.pips[i]['x'] = x + self.pipVx
            pass

        if self.pips[0]['x'] < 0:
            self.get_next_pip()

        self.fps = self.fps + 1
        pygame.display.flip()
        pygame.display.update()
        #FPSCLOCK.tick(FPS)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return terminal, reward, image_data
        pass


    def get_random_pips(self):
        speed_list = [0, 0]
        speed_list = speed_list +  [max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)]
        speed_list = speed_list +  [max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)]
        speed_list = speed_list +  [max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)]
        pip0 = {'x':PIP_INTER * 0, 'speed':speed_list[0]}
        pip1 = {'x':PIP_INTER * 1, 'speed':speed_list[1]}
        pip2 = {'x':PIP_INTER * 2, 'speed':speed_list[2]}
        pip3 = {'x':PIP_INTER * 3, 'speed':speed_list[3]}
        pip4 = {'x':PIP_INTER * 4, 'speed':speed_list[4]}

        return [pip0, pip1, pip2, pip3, pip4]
        pass

    def get_next_pip(self):
        self.pips = self.pips[1:]
        new_speed = self.pips[-1]['speed'] + max(min(19,round( (random.random() * MAX_ACC * 2 - MAX_ACC) * CONTROL_STEP, 2)) ,0)
        new_x = self.pips[-1]['x'] + 100
        self.pips = self.pips + [{'x':new_x, 'speed':new_speed}]
