from graphics import *
import numpy as np

# returns euclidian distance between two points
def dist(xy1,xy2):
    return ((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)**0.5

# superclass representing a graphical object
class G_Obj(object):
    def __init__(self,x,y,shape):
        self.x = x
        self.y = y
        self.shape = shape

    def move(self,dx,dy):
        self.shape.move(dx,dy)
        self.x+=dx
        self.y+=dy
        if self.__class__ == Actor:
            self.points[0]+=dx
            self.points[1]+=dy

    def draw(self,win):
        self.shape.draw(win)

    def undraw(self):
        self.shape.undraw()

# player character class that implements G_Obj
class Actor(G_Obj):
    def __init__(self,x,y,r,color='white'):
        shape = Circle(Point(x,y),r)
        shape.setFill(color)
        shape.setOutline(color)
        G_Obj.__init__(self,x,y,shape)
        self.r = r

        xpoints = []
        ypoints = []
        for xx in range(x-r,x+r+1):
            for yy in range(y-r,y+r+1):
                if dist((xx,yy),(x,y)) <= r:
                    xpoints.append(xx);ypoints.append(yy)
        self.points = [np.array(xpoints,np.int16),np.array(ypoints,np.int16)]
                    
    # test if the actor object (circle) intersects other objects
    def intersects(self,other):
        # if the type is a rectangular barrier, check to see if any points on the barrier's perimeter is within the circle's radius
        if type(other) == Barrier:
            # check top and bottom
            for x in range(other.x,other.x+other.width):
                for y in [other.y,other.y+other.height]:
                    if dist([x,y],[self.x,self.y]) < self.r:
                        return True; break
            # check left and right
            for y in range(other.y,other.y+other.height):
                for x in [other.x,other.x+other.width]:
                    if dist([x,y],[self.x,self.y]) < self.r:
                        return True; break
            return False

# rectangular barrier obstacle class that implements G_Obj
class Barrier(G_Obj):
    def __init__(self,x,y,width,height,color='white'):
        shape = Rectangle(Point(x,y),Point(x+width,y+height))
        shape.setFill(color)
        shape.setOutline(color)
        G_Obj.__init__(self,x,y,shape)
        self.width,self.height = width,height
# class containing physical quantities for the game and methods to update these; DOES NOT MOVE OBJECTS
class Physics(object):
    # all physical quantities are in pixels and frames (i.e. acceleration is pixels/frame^2)
    def __init__(self,vx,vy,g,vj):
        self.velocity_x = vx # horizontal speed (perceived speed of the player but actually the speed of the barriers)
        self.velocity_y = vy # initial vertical speed of the player
        self.gravity = g # downwards acceleration due to gravity (this should be a positive number)
        self.jump_boost = vj # increase in player vertical velocity due to jumping

    # adds the jump velocity to the current y-velocity
    def jump(self):
        self.velocity_y -= self.jump_boost

    # updates quantities per frame
    def tick(self):
        self.velocity_y += self.gravity

    



        
    

    
