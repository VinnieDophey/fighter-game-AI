# fighter-game-AI
I trained an AI to play a simple top-down fighter game I created!
If reading this on GitHub, you will not be able to see the gifs/demo stuff.

What does my Pygame AI program do?
My program allows a user to try out an AI that I’ve created which plays a simple top-down fighter game, where red enemies fly around and the blue player (controlled by AI) flies around with the goal of shooting down the red bandits.

Goal: Highest score (kill count) Lose if: out of rounds or out of bounds

Goals of this project
I have three main goals of this project:
	•	Learn what is machine learning and how it works With this project, I have gained a general understanding of how this all works, ranging from neural networks to crucial variables such as learning rate, gamma, and epsilon. Mathematical concepts used in ML such as linear algebra and calculus are outside my scope, but I have gained a general understanding on what their crucial role is in ML.
	•	AI Creation by using Python The demand for AI is increasing every year, as this is an emerging technology. This was an incentive for me to try creating an AI, as it would significantly enrich my understanding and help me explore if I would like to do this as a living.
	•	Document my Findings This project was overall a great experience for me, and I would like to document everything.  Since this is my first attempt at creating an AI, I would like to learn how I could do better in future projects.

How to run my program?
Step 1: Download the folder 
Step 2: Install packages
On Windows, one can install the following packages VSCode using pip3:
pip3 install xyz:

pygame
math
random
numpy
torch
collections
os
datetime
matplotlib
IPython

On Mac, some packages cannot be installed globally, so a virtual environment must be created: 
python3 -m venv ~/py_envs
source ~/py_envs/bin/activate
python3 -m pip install xyz

Alternatively, one can use an application like Thonny with Python Package Index (PyPI), and all that needs to be done is to go to the installation page and search up the libraries that must be installed.

Step 3: Run Agent.Py
On Mac, if using a virtual environment, one must enter “python agent.py” in the terminal, otherwise the packages will not be applied and an error message will appear.
The program may take half a minute to start up on the first run.

Once the program is ran, the user will be able to choose the following options in the terminal:
Downgrading resolution - The user will be prompted to downgrade to 1000 x 850 to meet with the final project requirements, however this project was built and trained on a 1920 x 1080 resolution so models may not perform as well with a lower resolution, and training new models will become more difficult as the AI has less room to turn due to smaller screens.
	•	Hidden layer, yes or no
	•	Loading model, yes or no
	•	If yes, it allows the user to choose a model that was found in the directory to load
	•	If yes, it also allows the user to set how many games had the model the user choose played (affects epsilon)
Cheat Code: Load the best models I’ve seen  (save the hassle of running the program for hours)
In the model folder, there are two files, one with the best model using hidden layers, and one with the best model without hidden layers. 

To run the best model  without hidden layers, do the following: Prompt to downgrade res: You may choose whichever, but “n” is better. Prompt for hidden layer: Answer “n” Prompt for loading model: Answer “y” Prompt for model selection: Choose the number that corresponds with !Best_No_Hidden_Model.pth Prompt for how many games played: Answer any integer above 11.

To run a trained model with hidden layers, do the following:
Prompt to downgrade res: You may choose whichever, but “n” is better.
Prompt for hidden layer: Answer “y” Prompt for loading model: Answer “y” Prompt for model selection: Choose the number that corresponds with !Best_Hidden_Model.pth Prompt for how many games played: Answer 56 (this model was only trained for 56 games, and still has lots of room for learning!)
I will talk about the limitations of each option later on.

Known “Bugs”:
If you receive a Runtime Error message, this means the wrong model was loaded in. A model with a hidden layer CAN NOT be loaded in a no-hidden layer setting and vice versa.

If the program says “no models found” while trying to load a model, it means two things:
	•	You may have deleted the model folder
	•	You may have run the program from another folder than the game folder. Ensure that the directory is  A:\bcdefg\hijk\Final_Game_Pygame_AI> 

Program Features - How it works, Findings, & Limitations
My program uses Machine Learning, which basically means that the AI learns strategies on its own with a ML algorithm (using Pytorch library).

This means that there isn’t an explicit algorithm telling the AI to “turn right! Turn left!” but there are rewards that are given to the AI to help the AI to make meaningful connections between actions and results.

The AI’s goal is to get the highest reward possible. With each frame and game, the AI adjusts weights in each neuron to optimize its strategy to get the highest reward. 


 My program, as mentioned beforehand, allows the user to pick if a hidden layer would be present. I will explore the advantages and limitations of the two options.  1. With a hidden layer
The AI learning is slow, but eventually after about an hour of running, the AI would learn the most optimal playstyle that it has discovered. 

Here’s one example where it spins around shooting anyone that spawns: 

*note that the lines are mainly for debugging and facilitate spotting the enemy for the watchers

Here’s another example with a still enemy, where it learnt to turn and shoot straight at the enemy.


Advantages:
	•	The AI is able to learn better strategies to maximize reward profits, in turn getting more score than without a hidden layer in the long run
	•	Pretty much guarantees good results after training it for a while
	•	Great for advanced applications (i.e. Autopilot or LLM’s)
	•	As you can see, results are always upward-trending: 
Limitations
	•	The AI tries testing new things through epsilon and dropout rate (if applicable) which may cause inconsistent performance
	•	It takes a long time to train ( >1 hour to get good results)
 2. Without a hidden layer
Without a hidden layer, the AI is able to exhibit algorithmic-like behaviours with only little exploration.  
Here is a near-perfect case scenario, where the AI has learned the “correct” algorithmic behavior. However, it is unable to learn to predict where enemies will be (shooting ahead) causing it to waste loads of ammo per kill, thus ending the round much earlier than its hidden layer counterpart.  

Advantages:
	•	The AI is able to have very consistent performance after the learning period
	•	Learns very quickly (only a dozen exploration games needed instead of the long 180 games epsilon gradient used for the hidden layered version)
Limitations:
	•	The AI never tries new things, so having a consistent performance is a double edge sword, as being consistently bad is no good.
	•	Some examples: many times when I’ve run it, the AI was only turning one direction instead of both as seen in the gif above. 
	•	Great for simple things, but an algorithm would be better for simple things. 
Here’s an example of it just being bad at turning left:
Unfortunately, this model will not get any better than this due to the fact that it has no hidden layers.

Additional features for users to play around with
You may have noticed that in the videos there are lines. These lines can be toggled by pressing the lowercase [L] key  Additionally, the learning graph that is also seen above can be toggled by pressing the lowercase [G] key. Note that this graph only shows up/updates once a game, so you must wait till the current game ends to see a graph.  The WASD SPACE keys are still fully functional, and allow a manual override of the AI’s actions by the player.  A = Left
D = Right W = Throttle Up
S = Throttle Down SPACE = Shoot
