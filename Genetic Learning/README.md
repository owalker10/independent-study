# Genetic Learning

After having limited success with using a Q-learning algorithm, I decided to try evolutionary learning

EL algorithms use principles of evolution (breeding, mutation, natural selection, fitness, etc.) to evolve an agent through generations

<br/>

<img src="https://cdn-images-1.medium.com/max/1600/1*HP8JVxlJtOv14rGLJfXEzA.png" width="275" height="200" />


## Progress:
- refitted `nn.py` to be used with EL (removed backpropagation, added mutation and crossover functions)
- created `species.py`, which holds classes that create the species structure
- created `game.py`, which is a version of Flappy Ball that can be played in a window or simulated without graphics
- created `evolution.py`, which simulates evolution of the player without graphics, and works pretty well! (Check out the mp4 clip)
- created `view_players.py` to watch trained AI play in a game window (complete with spooky textures for 800x600 versions)
    - though this requires a trained and pickled population file from `evolution.py`
    - you can open `EL Clip.mp4` to watch a recorded clip of an agent playing, instead
- continuing to train agents on different versions of Flappy Golf (speed, window size, etc.) and different training parameters
- added Halloween skins for maximum spooky fun!
- created `viz_brain.py` to draw a trained player "brain" (neural network) to a window

<br/>

<img src="https://github.com/owalker10/independent-study/blob/master/Genetic%20Learning/EL%20Gif.gif" width="300" height="225" />

<br/>

<img src="https://github.com/owalker10/independent-study/blob/master/Genetic%20Learning/EL%20Gif%20Spooky.gif" width="400" height="300" />

Demonstration of the EL algorithm in action!
