from graphics import *
from game_store import *
import time
from random import randint

'''
FLAPPY BALL GAME SCRIPT

Running this script plays the flappy ball game

INSTRUCTIONS:
- You are in control of the ball on the left side of the screen
- Press SPACE to jump
- Your goal is to go as far as you can, while avoiding running into the barriers

CONTROLS:
Press SPACE to start the game upon startup or after dying
Press Q mid-game to end the game
Press Q while the game is ended to exit the window

Credit to Zelle for the graphics library: http://mcsp.wartburg.edu/zelle/python/graphics.py
'''

WIDTH,HEIGHT = 800,600

FRAMERATE = 30

gravity = 1 # pixels/frame^2
speed = 4 # pixels/frame
init_y_velocity = 1 # pixels/frame
jump_speed = 20 # pixels/frame


barrier_prob = FRAMERATE * 1 # barriers should spawn at an average of 1 every n seconds
barrier_width = 75


win = GraphWin('Game',WIDTH,HEIGHT)
win.setBackground('Black')

time_text = Text(Point(50,50),'0.00')
time_text.setTextColor('white')

player = None

physics = None

barriers = []




def draw_all():
    player.draw(win)
    for barrier in barriers:
        barrier.draw(win)

    time_text.draw(win)

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
    player = Actor(150,round(HEIGHT/2),25)
    barriers.clear()

    time_text.setText('0.00')
    
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

    time_text.undraw()



def play():

    start_time = 0

    while True:
        restart()
        key = win.getKey()
        if key == 'q': exit()

        start_time = time.time()
        
        while key != 'q':
            key = win.checkKey()

            barrier_spawn()
            
            if key == 'space': physics.jump()
            frame()

            if check_loss():
                break

            time_text.setText(str(round(time.time()-start_time,2)))
            
            update(FRAMERATE)
            

        clear()

if __name__ == '__main__':
    play()


