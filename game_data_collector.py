from graphics import *
from game_store import *
import time
from random import randint
import numpy as np
import h5py
import math

'''
GAME DATA COLLECTOR SCRIPT

Running this script plays the flappy ball game. User input is collected, pickled, and stored to be used for analysis

This script saves the following data as a single one-dimensional vector (each data point is one frame):

- each pixel of the screen (1 for white, 0 for black), linearized by moving left to right then top to bottom
    - because this is a lot of data (too much), the pixels are "blurred" by summing up ones and zeros in 2x2 squares,
    this cuts the data size to 25%
- the current y velocity of the avatar
- whether or not the avatar jumped (pressed "spacebar")


CONTROLS:
Press SPACE to start the game upon startup or after dying
Press Q mid-game to end the game
Press Q while the game is ended to exit the window

Credit to Zelle for the graphics library: http://mcsp.wartburg.edu/zelle/python/graphics.py
'''

WIDTH,HEIGHT = 400,300 # game screen width and height

FRAMERATE = 20

gravity = 1 # pixels/frame^2
speed = 5 # pixels/frame
init_y_velocity = 0 # pixels/frame
jump_speed = 15 # pixels/frame


barrier_prob = FRAMERATE * 1 # barriers should spawn at an average of 1 every n seconds
barrier_width = 50 # width of barriers in pixels


win = GraphWin('Game',WIDTH,HEIGHT)
win.setBackground('Black')


player = None # player object

physics = None # physics object

barriers = [] # list of barriers on screen

data = [[],[],[]] # list of pixel matrices, player velocity, and whether or not the player jumped



# draw objects on screen (this only needs to be called once per game)
def draw_all():
    player.draw(win)
    for barrier in barriers:
        barrier.draw(win)

# called every frame, has random chance to spawn barriers
def barrier_spawn():
    i = randint(1,barrier_prob)
    if i == 1:
        # check to make sure there are 3 barrier widths of space before spawning a new barrier
        if barriers == [] or (barriers != [] and barriers[-1].x < WIDTH - barrier_width*3):
            # randomly distributed height between 25% and 60% of the screen
            height = randint(round(HEIGHT*0.25),round(HEIGHT*0.6))
            # 50/50 chance to spawn top barrier or bottom barrier
            if randint(1,2) == 1:
                barrier = Barrier(WIDTH,HEIGHT-height,barrier_width,height)
            else:
                barrier = Barrier(WIDTH,0,barrier_width,height)
            barriers.append(barrier)
            barrier.draw(win)



# restarts variables, called after player dies
def restart():
    global player,physics,start_time
    physics = Physics(speed,init_y_velocity,gravity,jump_speed)
    player = Actor(150,round(HEIGHT/2),15)
    barriers.clear()

    draw_all()

# called every frame, iterates values and move objects
def frame():
    player.move(0,physics.velocity_y)
    physics.tick()

    for barrier in barriers:
        barrier.move(-physics.velocity_x,0)
        if barrier.x+barrier.width < 0:
            barrier.undraw()
            barriers.remove(barrier)

# checks to see if player has collided and died
def check_loss():
    if player.y + player.r > HEIGHT or player.y < 0:
        return True
    for barrier in barriers:
        if player.intersects(barrier):
            return True
    return False

# clears playing window
def clear():
    player.undraw()
    for barrier in barriers:
        barrier.undraw()

# add collected data to lists
def collect(win,vy,jump):
    global data

    pixels = get_pixels()

    pixels = blur(pixels)

    data[0].append(pixels)
    data[1].append(vy)
    data[2].append(jump)

# creates a WIDTH by HEIGHT numpy matrix of pixels on screen by checking all points where player and barriers exist
def get_pixels():
    pixels = np.zeros((WIDTH,HEIGHT),np.int8)

    pixels[player.points[0]-1,player.points[1]-1] = 1

    for b in barriers:
        p1,p2 = b.shape.getP1(),b.shape.getP2()
        pixels[int(p1.getX())-1:min(int(p2.getX()),WIDTH),int(p1.getY())-1:int(p2.getY())] = 1

    return pixels

# "blurs" pixels of a frame by taking 2x2 squares of pixels and summing them
def blur(frame):
    a = np.zeros((frame.shape[0]//2,frame.shape[1]//2))
    for x in range(0,frame.shape[0],2):
        for y in range(0,frame.shape[1],2):
            a[x//2,y//2] = np.sum(frame[x:x+2,y:y+2])
    return a


# game loop
def play():
    global data

    start_time = 0

    while True:
        restart()

        # quit if key pressed is "q"
        key = win.getKey()
        if key == 'q': break

        start_time = time.time()

        while key != 'q':
            vy = physics.velocity_y
            jump = 0
            key = win.checkKey()

            barrier_spawn() # chance to spawn barriers

            if key == 'space': # if space is pressed, jump
                jump = 1
                physics.jump()

            collect(win,vy,jump) # collect values

            frame() # iterate the frame

            if check_loss(): # check for loss, if so, break
                break

            update(FRAMERATE) # controls the framerate


        clear() # clears the screen for a restart


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
