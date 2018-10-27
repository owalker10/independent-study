from graphics import *
from game_store import *
import time
from random import randint
import numpy as np
import h5py
import math
import nn
from sklearn.preprocessing import MinMaxScaler

'''
REINFORCEMENT LEARNING SCRIPT

Running this script starts a reinforcement learning algorithm version of the Flappy Ball game.
No user input is required. The player is controlled by a pair of neural networks that decide if the player should jump.
The neural networks are trained by an algorithm called "backpropogation" which involves a lot of multivariable calculus and linear algebra.
The player doesn't ever learn how to play particularly well, so I'm currently implementing a different learning algorithm.

Credit to Zelle for the graphics library: http://mcsp.wartburg.edu/zelle/python/graphics.py
'''

remove_skins() # take off the Halloween skins I made

WIDTH,HEIGHT = 400,300 # window dimensions

FRAMERATE = 20

gravity = 1 # pixels/frame^2
speed = 5 # pixels/frame
init_y_velocity = 0 # pixels/frame
jump_speed = 5 # pixels/frame


barrier_prob = FRAMERATE * 1 # barriers should spawn at an average of 1 every n seconds
barrier_width = 50 # width of barriers in pixels


win = GraphWin('Game',WIDTH,HEIGHT) # window object
win.setBackground('Black')


player = None # player object

physics = None # physics object

barriers = [] # list of barriers on screen

data = [[],[],[],[]] # data collected from current life, used to train model

rewards = [] # list of rewards per frame (the player seeks to maximize this)
gamma = 0.98 # decay factor for reward

e0 = 0.10 # probability of choosing to jump when NN decides not to

e1 = 0.30 # probability of choosing not to jump when NN decides to jump

text = Text(Point(100,50),'e0: '+str(e0)+' e1: '+str(e1));text.setTextColor('white');text.draw(win)

#model0 = nn.NeuralNetwork([WIDTH*HEIGHT//4+2,300,1],['relu','linear'],0.02,'SSE',kind='batch')
#model1 = nn.NeuralNetwork([WIDTH*HEIGHT//4+2,300,1],['relu','linear'],0.02,'SSE',kind='batch')

'''
instantiate neural networks with the following parameters:
    - layers: number of neurons per layer of the networks
    - activations: the activation functions for every layer
    - training step size: the fraction of the cost gradient to adjust by every training iteration
    - cost function: the cost function to optimize (SSE stands for Sum of Squares Error)
    - kind: the kind of neural network to use (batch means that there are multiple data points being trained on per iteration)
'''
model0 = nn.NeuralNetwork([WIDTH*HEIGHT//4+2,100,100,1],['relu','relu','linear'],0.01,'SSE',kind='batch')
model1 = nn.NeuralNetwork([WIDTH*HEIGHT//4+2,100,100,1],['relu','relu','linear'],0.01,'SSE',kind='batch')

scaler = MinMaxScaler() # scales values to [0,1] for neura network

# create a scaler that scales velocity values between approximately 0 and 1
with h5py.File('vys.h5','r') as f:
    v_data = np.array(f['vys'])
    bounds = np.array([v_data.min()-2*v_data.std(),v_data.max()+2*v_data.std()]).reshape(-1,1)
    scaler.fit(bounds)


# draw objects on screen (this only needs to be called once per game)
def draw_all():
    player.draw(win)
    for barrier in barriers:
        barrier.draw(win)

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
    player = Actor(150,round(HEIGHT/2),15)
    barriers.clear()

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

# adds the jump value (whether the player jumped that frame) to the data array
def collect_jump(jump):
    global data
    data[2].append(jump)

# adds the pixels, y velocity, and y position to the data array
def collect(win,vy,y_pos):
    global data

    pixels = get_pixels()

    pixels = blur(pixels)

    data[0].append(pixels)
    data[1].append(np.asscalar(scaler.transform(vy)))
    data[3].append(y_pos)


# get all white pixels in the current frame
def get_pixels():
    pixels = np.zeros((WIDTH,HEIGHT),np.int8)

    pixels[player.points[0]-1,player.points[1]-1] = 1

    for b in barriers:
        p1,p2 = b.shape.getP1(),b.shape.getP2()
        pixels[int(p1.getX())-1:min(int(p2.getX()),WIDTH),int(p1.getY())-1:int(p2.getY())] = 1

    return pixels

# blurs the pixels to reduce the size of the pixel matrix
def blur(frame):
    # will initially produce values 0-4
    a = np.zeros((frame.shape[0]//2,frame.shape[1]//2))
    for x in range(0,frame.shape[0],2):
        for y in range(0,frame.shape[1],2):
            a[x//2,y//2] = np.sum(frame[x:x+2,y:y+2])
    return a/4

# calculated decayed reward at every frame given a list of rewards per frame
def decayed_reward(R):
    return sum([R[::-1][n]*gamma**n for n in range(len(R))])/50+1

def qbeg(q): #q-based e values for e-greedy decision making, bounded between e = 0.01 and 0.3
    # turns out, these don't help so we're not using them
    q_min,q_max = 0.0,-1.0
    e_min,e_max = 0.01,0.3
    e = ((e_max-e_min)/(q_max-q_min))*(q-q_min) + e_min
    return max(e_min, min(e_max, e))

# train the neural networks on the actual reward (Q) values
def train(model0,model1):
    global data,rewards
    batch0 = [[],[]]
    batch1 = [[],[]]
    # add all the data to the non-jump (0) NN input data or the jump NN (1) input data
    for n,dat in enumerate(zip(data[0],data[1],data[2],data[3])):
        pix,v,jump,y_pos = dat
        Q = decayed_reward(rewards[n:])
        #print('Q-values',Q)
        x = np.append(pix.ravel(),[v,y_pos])
        if jump == 0:
            batch0[0].append(x)
            batch0[1].append(Q)
        else:
            batch1[0].append(x)
            batch1[1].append(Q)


    batch0[0] = np.array(batch0[0]).T
    batch0[1] = np.array(batch0[1])

    batch1[0] = np.array(batch1[0]).T
    batch1[1] = np.array(batch1[1])

    # train both networks and print cost
    if np.any(np.isin(0,data[2])): print('0 cost',model0.backpropogation(batch0[0],batch0[1]))
    if np.any(np.isin(1,data[2])): print('1 cost',model1.backpropogation(batch1[0],batch1[1]))

    # reset data and rewards
    data = [[],[],[],[]]
    rewards = []

# based on the estimated rewards, choose to jump or not jump
def decide(win,vy,y_pos):
    global e0,e1
    collect(win,vy,y_pos)
    x = np.append(data[0][-1].ravel(),[data[1][-1],data[3][-1]]).reshape(-1,1)
    Q0 = model0.forward_propogation(x)[0][-1]
    Q1 = model1.forward_propogation(x)[0][-1]

    # this code doesn't help...
    '''
    e0 = qbeg(Q0)
    e1 = qbeg(Q1)
    if type(e0) is not float or type(e1) is not float:
        e0,e1 = float(e0),float(e1)

    text.setText('e0: '+str(round(e0,2))+' e1: '+str(round(e1,2)))
    '''


    p = np.random.rand()
    #print('Q0, Q1',Q0,Q1)

    #choose action based on e-greedy policy (random chance to choose non-optimal option)
    if Q1>Q0 and p > e1:
        return 'space'

    elif Q1<Q0 and p < e0:
        return 'space'

    else:
        return None

# game loop
def play():
    global data,e0,e1

    start_time = 0
    scores = []

    while True: # whole game loop
        restart()



        start_time = time.time()

        scores.append(0)
        while True: # duration of one player life


            key = win.checkKey()

            vy = physics.velocity_y
            jump = 0
            key = decide(win,vy,player.y/HEIGHT) # use the NN's to decide to jump or not
            #print(key)

            barrier_spawn() # chance to spawn barriers

            if key == 'space':
                jump = 1
                physics.jump()

            collect_jump(jump)

            frame()

            if check_loss():
                rewards.append(-100)

                # change likelihood to randomly jump/not jump based on if the player died at the top or bottom of the screen
                if player.y < HEIGHT/2:
                    e0-=0.02;e1+=0.02
                else:
                    e0+=0.02;e1-=0.02

                # change e values on screen
                text.setText('e0: '+str(round(e0,2))+' e1: '+str(round(e1,2)))


                break

            # add reward of 3 every frame the player doesn't die
            rewards.append(3)

            scores[-1]+=1

            #update(FRAMERATE)

        print('score',scores[-1])

        # train the models after the player dies
        train(model0,model1)





        clear() # clears the screen for a restart

        if win.checkKey() == 'q': break


    # to avoid memory errors, we save the pixel data in chunks of 400 frames
    '''
    f = h5py.File('data2.h5','w')
    for i in range(len(data[0])//400+1):
        chunk = data[0][0:min(len(data[0]),400)]
        f.create_dataset('pixels'+str(i),data=np.array(chunk,np.int8))
        del data[0][0:min(len(data[0]),400)];del chunk
    del data[0]
    f.create_dataset('vys',data=np.array(data[0],np.int8)); del data[0]
    f.create_dataset('jump',data=np.array(data[0],np.int8)); del data[0]
    f.close()
    '''

    np.save('rl_scores.npy',np.array(scores))




    exit()

if __name__ == '__main__':
    play()
