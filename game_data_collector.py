from graphics import *
from game_store import *
import time
from random import randint
import numpy as np
import h5py

'''
GAME DATA COLLECTOR SCRIPT

Running this script plays the flappy ball game. User input is collected, pickled, and stored to be used for machine learning

This script saves the following data as a one-dimensional vector (each data point is one frame):
each pixel of the screen (1 for white, 0 for black), linearized by moving left to right then top to bottom
the current y velocity of the avatar
    

CONTROLS:
Press SPACE to start the game upon startup or after dying
Press Q mid-game to end the game
Press Q while the game is ended to exit the window

Credit to Zelle for the graphics library: http://mcsp.wartburg.edu/zelle/python/graphics.py
'''

WIDTH,HEIGHT = 400,300

FRAMERATE = 20

gravity = 1 # pixels/frame^2
speed = 5 # pixels/frame
init_y_velocity = 0 # pixels/frame
jump_speed = 15 # pixels/frame


barrier_prob = FRAMERATE * 1 # barriers should spawn at an average of 1 every n seconds
barrier_width = 50


win = GraphWin('Game',WIDTH,HEIGHT)
win.setBackground('Black')


player = None

physics = None

barriers = []

data = [[],[],[]]




def draw_all():
    player.draw(win)
    for barrier in barriers:
        barrier.draw(win)

def barrier_spawn():
    i = randint(1,barrier_prob)
    if i == 1:
        if barriers == [] or (barriers != [] and barriers[-1].x < WIDTH - barrier_width*3):
            height = randint(round(HEIGHT*0.25),round(HEIGHT*0.6))
            if randint(1,2) == 1:
                barrier = Barrier(WIDTH,HEIGHT-height,barrier_width,height)
            else:
                barrier = Barrier(WIDTH,0,barrier_width,height)
            barriers.append(barrier)
            barrier.draw(win)
            
            


def restart():
    global player,physics,start_time
    physics = Physics(speed,init_y_velocity,gravity,jump_speed)
    player = Actor(150,round(HEIGHT/2),15)
    barriers.clear()
    
    draw_all()
    

def frame():
    player.move(0,physics.velocity_y)
    physics.tick()
    
    for barrier in barriers:
        barrier.move(-physics.velocity_x,0)
        if barrier.x+barrier.width < 0:
            barrier.undraw()
            barriers.remove(barrier)


def check_loss():
    if player.y + player.r > HEIGHT or player.y < 0:
        return True
    for barrier in barriers:
        if player.intersects(barrier):
            return True
    return False

def clear():
    player.undraw()
    for barrier in barriers:
        barrier.undraw()

def collect(win,vy,jump):
    global data

    pixels = get_pixels()

    data[0].append(pixels)
    data[1].append(vy)
    data[2].append(jump)


def get_pixels():
    pixels = np.zeros((WIDTH,HEIGHT),np.int8)
    
    pixels[player.points[0]-1,player.points[1]-1] = 1

    for b in barriers:
        p1,p2 = b.shape.getP1(),b.shape.getP2()
        pixels[int(p1.getX())-1:min(int(p2.getX()),WIDTH),int(p1.getY())-1:int(p2.getY())] = 1

    return pixels


def play():
    global data

    start_time = 0

    while True:
        restart()
        
        key = win.getKey()
        if key == 'q': break

        start_time = time.time()
        
        while key != 'q':
            vy = physics.velocity_y
            jump = 0
            key = win.checkKey()

            barrier_spawn()
            
            if key == 'space':
                jump = 1
                physics.jump()

            collect(win,vy,jump)
                
            frame()

            if check_loss():
                break
            
            update(FRAMERATE)
            

        clear()


    # to avoid memory errors, we save the pixel data in chunks of 400 frames
    f = h5py.File('data.h5','w')
    for i in range(len(data[0])//400+1):
        chunk = data[0][0:min(len(data[0]),400)]
        f.create_dataset('pixels'+str(i),data=np.array(chunk,np.int8))
        del data[0][0:min(len(data[0]),400)];del chunk
    del data[0]
    f.create_dataset('vys',data=np.array(data[0],np.int8)); del data[0]
    f.create_dataset('jump',data=np.array(data[0],np.int8)); del data[0]
    f.close()
    
    
        

    exit()

if __name__ == '__main__':
    play()


