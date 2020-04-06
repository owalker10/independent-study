# independent-study

Repository of ongoing work in Comp Sci Independent Study class. Current objective is to use reinforcement learning to train a program to play a small game.

## Progress:
- created "Flappy Ball" game, located at `game.py`
- created script that builds multi-layer perceptron neural networks (linearly layered NN) with a variety of parameters (number of neurons at nth layer, activation functions, etc.)
- built a test script for the Neural Network that predicts wine quality (`nn_regression_test.py`)
- created script that using Q-learning to train a player to play Flappy Ball (`game_rl.py`). Works, but not very well.
- started testing evolutionary learning algorithms as a substitute for Q-learning (everything located in `Genetic Learning` subdirectory). Already works much better.
- Retextured the 800x600 versions of Flappy Ball in Halloween skins because I can.
- Continued training players with Evolutionary Learning, trained players are now very successful.

<br/>

<img src="https://github.com/owalker10/independent-study/blob/master/Genetic%20Learning/EL%20Gif.gif" width="300" height="225" />

<br/>

<img src="https://github.com/owalker10/independent-study/blob/master/Genetic%20Learning/EL%20Gif%20Spooky.gif" width="400" height="300" />

Demonstration of the EL algorithm in action!

<br/><br/>

<img src="https://github.com/owalker10/independent-study/blob/master/Genetic%20Learning/nn brain.png" width="600" height="400" />


This is what a trained neural network looks like.
