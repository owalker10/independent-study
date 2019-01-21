import species
import game
import pickle

'''
This script takes an evolved population and sorts its generations by average fitness over 8 tries of the most fit player
Results are stored as a sorted list of tuples, pickled, and saved to the disk
'''

species.X = 150
species.Y = 300
species.R = 25
game.jump_speed = 20
game.WIDTH,game.HEIGHT = 800,600
game.speed = 6
game.barrier_prob = 10
game.barrier_width = 75
game.vision_rows=game.rows(10)


# load and player and have it play the game, keeping track of fitness scores

with open('population5.pkl','rb') as file:
    population = pickle.load(file)

avg_fs = {}
for i in range(population.gen+1):
    fs = []
    for n in range(8):
        player = population.generations[i].players[-1]
        fs.append(game.play(player))
    avg_fs[i] = sum(fs)/len(fs)
    print(i)

ordered = sorted(avg_fs.items(), key=lambda kv: kv[1])

with open('ordered_avg_fitness5.pkl','wb') as file:
    pickle.dump(ordered,file)

print(ordered[-1][0])
