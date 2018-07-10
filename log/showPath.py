import pygame
import math
import argparse

ROAD_WIDTH = 9.0
IMG_WIDTH = 860
IMG_HEIGHT = 800
PIXEL_DENSITY = 2
START_MAP_X = 20
START_MAP_Y = 20
FPS = 25



def main():
    argparser = argparse.ArgumentParser(description="show predefined path")
    argparser.add_argument('-n','--path_number',default='all')
    argparser.add_argument('-d','--dir',default='./')
    road_left,road,road_right = load_routes()
    print(road_left[0])
    print(road[0])
    print(road_right[0])

    #init
    pygame.init()
    screen = pygame.display.set_mode( (IMG_WIDTH, IMG_HEIGHT), 0, 32)
    pygame.display.set_caption('show way path')
    fpsClock = pygame.time.Clock()
    screen.fill((0,0,0))

    map_points = []
    map_points_left = []
    map_points_right = []
    #edge_points = []
    for road_point in road:
        map_points.append(( int((road_point[0] + START_MAP_X) * PIXEL_DENSITY), \
                            int((road_point[1] + START_MAP_X) * PIXEL_DENSITY)  ))
    for road_point_left in road_left:
        map_points_left.append(( int((road_point_left[0] + START_MAP_X) * PIXEL_DENSITY), \
                            int((road_point_left[1] + START_MAP_X) * PIXEL_DENSITY)  ))
    #edge_points = map_points_left
    for road_point_right in road_right:
        map_points_right.append(( int((road_point_right[0] + START_MAP_X) * PIXEL_DENSITY), \
                            int((road_point_right[1] + START_MAP_X) * PIXEL_DENSITY)  ))
        #edge_points = [( road_point_right[0],road_point_right[1] )] + edge_points
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            #pygame.draw.lines(screen, (255,255,255), True, map_points, int(ROAD_WIDTH * PIXEL_DENSITY) )
            #pygame.draw.polygon(screen, (255,255,255), map_points_right, 0 )
            #pygame.draw.polygon(screen, (0,0,0), map_points_left, 0 )
            #pygame.draw.polygon(screen, (255,255,255), edge_points, 0 )
            pygame.draw.lines(screen, (255,0,0), True, map_points, 25 )
            pygame.display.update()
            fpsClock.tick(FPS)

    finally:
        pygame.quit()
    
    pass


def load_routes():

    points = []
    points_left = []
    points_right = []

    def readfile(name):
        with open(name) as f:
            line = f.readline()
            while line:
                contents = line.split(',')
                if len(contents) == 3:
                    x,y,yaw = float(contents[0]),  float(contents[1]), float(contents[2])
                    points.append(( x,y,yaw ))
                    points_left.append(( x + ROAD_WIDTH/2 * math.sin(math.radians(yaw)), y - ROAD_WIDTH/2 * math.cos(math.radians(yaw)), yaw ))
                    points_right.append(( x - ROAD_WIDTH/2 * math.sin(math.radians(yaw)), y + ROAD_WIDTH/2 * math.cos(math.radians(yaw)), yaw ))
                    pass
                line = f.readline()

    n = '1.txt'
    readfile(n)
    for i in range(1,15):
        #load route i->j
        n = str(i) + '-' + str(i+1) + '.txt'
        readfile(n)
        #load route i+1
        n = str(i+1) + '.txt'
        readfile(n)

    return points_left, points, points_right
    pass



if __name__ == '__main__':

    main()

    pass
