from random import randint
from graphics import *

'''
This script simualates the Flappy Ball game, with or without the graphics window
'''

WIDTH,HEIGHT = 400,300

gravity = 1
speed = 5
jump_speed = 15

barrier_prob = 20 # barriers will spawn 1 in 20 frames
barrier_width = 50

# create boundaries for equal horizontal rows going across the screen
def rows(num_rows):
    return [[n,n+HEIGHT//num_rows-1] for n in range(1,HEIGHT//num_rows*(num_rows-1)+2,HEIGHT//num_rows)]

# horizontal sections of pixels, for each of these rows the player can see how close the closest barrier is
num_rows = 6
vision_rows = rows(num_rows)

win = None
framerate = 30 # only used if we're displaying players

def dist(xy1,xy2):
    return ((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)**0.5

class Barrier():
    def __init__(self,x,y,width,height,graphics):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.graphics = graphics

        if graphics:
            shape = Rectangle(Point(x,y),Point(x+width,y+height))
            shape.setFill('white')
            shape.setOutline('white')
            self.shape = shape
            self.shape.draw(win)

    def move(self,dx,dy):
        self.x += dx
        self.y += dy

        if self.graphics:
            self.shape.move(dx,dy)

    def undraw(self):
        self.shape.undraw()


class Physics():
    def __init__(self,vx,vy,g,vj):
        self.vx = vx
        self.vy = vy
        self.g = g
        self.vj = vj

    def jump(self):
        self.vy -= self.vj

    def tick(self):
        self.vy += self.g


# given the grouped rows of pixels as defined at the top of this file,
# find out how close the nearest barrier is to the player in each row
def look(barriers,x,r):
    s = x - r # left-most coordinate of player
    vision = [WIDTH-s]*len(vision_rows)

    for b in barriers:
        for n,row in enumerate(vision_rows):
            # if any pixel in the barrier is within the row and isn't behind the player:
            if (not (b.y > row[1] or b.y+b.height < row[0])) and b.x+b.width >= s:
                x = max(b.x,s) - s
                if x < vision[n]: vision[n] = x

    return vision

def tick(player,phys,barriers,graphics,fitness,f_text):
    player.move(0,phys.vy)
    phys.tick()

    if graphics:
        f_text.setText(str(fitness))

    for n,b in enumerate(barriers):
        b.move(-phys.vx,0)
        if b.x+b.width < 0:
            if graphics:
                barriers[n].undraw()
            del barriers[n]


def is_loss(player,barriers):
    if player.y + player.r > HEIGHT or player.y - player.r < 0:
        return True

    for b in barriers:
        if player.intersects(b): return True

    return False


def barrier_spawn(barriers,graphics):
    r = randint(1,barrier_prob)
    if r == 1:
        # if there are no barriers or the closest barrier is 3 barrier widths away:
        if barriers == [] or (barriers != [] and barriers[-1].x < WIDTH - barrier_width*3):
            # find a random height between 25 and 60% of screen height
            height = randint(round(HEIGHT*.25),round(HEIGHT*.6))
            if randint(1,2) == 1: # 50/50 chance to be on top or bottom
                barrier = Barrier(WIDTH,HEIGHT-height,barrier_width,height,graphics)
            else:
                barrier = Barrier(WIDTH,0,barrier_width,height,graphics)

            barriers.append(barrier)

def undraw(player,f_text,barriers):
    player.undraw()
    f_text.undraw()
    for b in barriers:
        b.undraw()


def play(player,graphics=False):
    barriers = []
    fitness = 0

    f_text=None
    if graphics:
        player.draw(win)
        f_text = Text(Point(50,50),str(fitness))
        f_text.draw(win)


    phys = Physics(speed,0,gravity,jump_speed)

    while True:
        fitness+=1
        vision = look(barriers,player.x,player.r)
        vision.extend([phys.vy/20,player.y/HEIGHT])
        jump = player.decide(vision,len(vision_rows))
        if jump: phys.jump()
        tick(player,phys,barriers,graphics,fitness,f_text)
        if is_loss(player,barriers):
            if graphics: undraw(player,f_text,barriers)
            break
        barrier_spawn(barriers,graphics)

        if graphics: update(framerate)

    return fitness
