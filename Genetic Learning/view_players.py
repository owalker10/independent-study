import species
import game
from graphics import *
import pickle

'''
This script allows us to load pickled brains, players, or populations
and view certain players as the play on the graphics window
'''

# new Player class, inherits player class from species and supports graphics functions
class Player(species.Player):
    def __init__(self,player):
        species.Player.__init__(self,player.num,player.imr,player.amr,player.io,player.hidden_layers,player.activations)
        self.brain = player.brain

        shape = Circle(Point(self.x,self.y),self.r)
        shape.setFill('white')
        shape.setOutline('white')
        self.shape = shape

    def draw(self,win):
        self.shape.draw(win)

    def undraw(self):
        self.shape.undraw()

    def move(self,dx,dy):
        super(Player,self).move(dx,dy)
        self.shape.move(dx,dy)

species.X = 150
species.Y = 300
species.R = 25
game.jump_speed = 20
game.WIDTH,game.HEIGHT = 800,600
game.speed = 6
game.barrier_prob = 30
game.barrier_width = 75
game.vision_rows=game.rows(6)


# load and player and have it play the game, keeping track of fitness scores

#population = None
with open('population2.pkl','rb') as file:
    population = pickle.load(file)

# we'll take the best player from the last generation of this population
pop_player = population.generations[-1].players[-1]

player = Player(pop_player)



game.win = GraphWin('Flappy Ball EL',game.WIDTH,game.HEIGHT)
#background = Image(Point(game.WIDTH//2,game.HEIGHT//2),'game_background.png'); background.draw(game.win)
game.win.setBackground('black')
game.framerate = 30
while True:
    fitness = game.play(player,graphics=True)
    print(fitness)
    player = Player(pop_player)
