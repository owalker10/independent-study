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

WIDTH,HEIGHT = 800,600 # window dimensions

FRAMERATE = 30

gravity = 1 # pixels/frame^2
speed = 6 # pixels/frame
#speed = 9
init_y_velocity = 1 # pixels/frame
jump_speed = 20 # pixels/frame


#barrier_prob = FRAMERATE * 1 # barriers should spawn at an average of 1 every n seconds
barrier_prob = 10
barrier_width = 75 # width of barriers in pixels


win = GraphWin('Game',WIDTH,HEIGHT) # window object
#win.setBackground('Black')
background = Image(Point(WIDTH//2,HEIGHT//2),'Genetic Learning/game_background.png'); background.draw(win)

time_text = Text(Point(50,50),'0.00') # text that displays time
time_text.setTextColor('white')

player = None # player object

physics = None # physics objects

barriers = [] # list of barriers on screen



# draw objects on screen (this only needs to be called once per game)
def draw_all():
    player.draw(win)
    for barrier in barriers:
        barrier.draw(win)

    time_text.draw(win)

# called every frame, has random chance to spawn barriers
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




# restarts variables, called after player dies
def restart():
    global player,physics,start_time
    physics = Physics(speed,init_y_velocity,gravity,jump_speed)
    player = Actor(150,round(HEIGHT/2),25)
    barriers.clear()

    time_text.setText('0.00')

    draw_all()


# called every frame, iterates values and moves objects
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
    if player.y + player.r > HEIGHT or player.y - player.r < 0:
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

    time_text.undraw()


# game loop
def play():

    start_time = 0

    while True:
        restart()
        key = win.getKey()
        if key == 'q': exit() # if Q is pressed, quit game

        start_time = time.time()

        #score = 0
        
        while key != 'q': # if Q is pressed, quit game
            #score+= 1
            
            key = win.checkKey()

            barrier_spawn() # chance to spawn barriers

            if key == 'space': physics.jump() # if space is pressed, jump
            frame() # iterate values and move objects

            if check_loss(): # check for loss, if so, break
                break

            time_text.setText(str(round(time.time()-start_time,2))) # update time text

            update(FRAMERATE) # controls frameraet

        #print(score)
        clear() # clears the screen for a restart

if __name__ == '__main__':
    play()
