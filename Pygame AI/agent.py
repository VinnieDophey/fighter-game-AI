"""
Summative - AI in PyGame
ICS3U-03
Vincent Pham
Agent file (connects game file with model)
History: Version 1 - May 30, 2025
"""

# BASICS - Imports
import torch
import random
import numpy as np
from collections import deque
from game import Game, SCREENX, SCREENY #direction and point from vid (?)
from model import Linear_QNet, QTrainer
from helper import plot #plot2
import os

# Define constants
MAX_MEMORY = 100_100
BATCH_SIZE = 1000
LR =  0.01


class Agent:
    
    def __init__(self):
        """
        Initializes the Agent
        This file initializes all the important variables, as well as asks the user some customization questions. 
        Args: None
        Returns: None
        """
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # Deque deletes older elements when bigger than max mem
        self.hiddenToggle = False
        hiddenInput = input("\nWelcome to my PyGame AI! \nWould you like the AI to have a hidden layer or no? (Y/N): ").lower()
        while hiddenInput != 'y' and hiddenInput != 'n':
            hiddenInput = input("Would you like the AI to have a hidden layer or no? (Y/N): ").lower()
        if hiddenInput == 'y':
            self.hiddenToggle = True
        
        self.model = Linear_QNet(1, 256, 3, self.hiddenToggle)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        
        # Since the agent init only runs once (when the user runs the program), I can make some user input here!
        self.loadModel()

    def loadModel(self):
        """
        Loads a model (sends it to the model.py file) if the user selects yes
        Args: None
        Returns: None
        """
        model_files = []
        model_dir = './model' # This is so than its less prone to mistake (varaibles auto complete)
        
        # Check if there are models 
        if not os.path.exists(model_dir):
            print(f"No model directory found at {model_dir}")
            return

        # Get list of model files
        for filename in os.listdir(model_dir):
            if filename.endswith('.pth'):
                model_files.append(filename)
        
        if not model_files:
            print("No model files found in model folder")
            return

        userInput = input("\nDo you want to load a pre-existing model? (Y/N): ").lower()
        while userInput != 'y' and userInput != 'n':
            print("\nInvalid input!")
            userInput = input("Do you want to load a pre-existing model? (Y/N): ").lower()
            
        if userInput == 'y':
            print("\nModels found in current directory:")
            for i, x in enumerate(model_files):
                print(f"{i+1}: {x}")
                
            model_num = input("Enter model number to load: ")
            while not(model_num.isdigit()) or not(int(model_num) in range(len(model_files)+1)):
                print("\nInvalid input! Please enter a valid model number")
                model_num = input("Enter model number to load: ")
                
            selected_model = model_files[int(model_num)-1]
            file_path = os.path.join(model_dir, selected_model)
            
            if self.model.load(file_path):
                print(f"Successfully loaded model: {selected_model}")
                epsilon_choose = input("\nHow many games played? (180 for full logic) \nThis affects epsilon, but the graph will not show this number (starts at 0): ")
                while not(epsilon_choose.isdigit()):
                    print("Please enter a valid integer!")
                    epsilon_choose = input("\nHow many games played? (180 for full logic) \nThis affects epsilon, but the graph will not show this number (starts at 0): ")
                self.n_games = int(epsilon_choose)
                
            else:
                print(f"Failed to load model: {selected_model}")
        print("Pygame has started running. Check alt tab to find it. Have fun watching my AI!")
        
    
    def get_state(self, game):
        """
        Gets the state of the game
        Args: game (class) - all of the variables in Game class
        Returns: state of the game (numpy array)
        """
        player = game.player
        state = [
            game.relativeAngle] 
            
        return np.array(state, dtype=np.float32)

    
    def remember(self, state, action, reward, next_state, done):
        """
        Remembers past actions, the results of the actions, and the reward,
        Args: 
        state (np array) - state of the game
        action (List) - what the AI did
        reward (Int) - The reward that the AI got for the action (in game.py reward is calculated)
        next_state (np array) - state of the game after the action 
        done (bool) - if the game is done or not (If True, it's done)
        Returns: None
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        """
        Training long term memory (through big array)
        Args: None
        Returns: None
        """
        # If not batch size yet it just adds to the sample. Otherwise, removes oldest memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples (with everything)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # Training (changing weights)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Training long term memory (through small array)
        Args: 
        state (np array) - state of the game
        action (List) - what the AI did
        reward (Int) - The reward that the AI got for the action (in game.py reward is calculated)
        next_state (np array) - state of the game after the action 
        done (bool) - if the game is done or not (If True, it's done)
        Returns: None
        """
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        """
        Gets the action that the AI chooses (or randomly chosen by epsilon)
        Args: 
        state (np array) - state of the game
        Returns: final_move (list) - A list with a length of 3, with one of them showing 1 instead of 0, indicating the AI action.
        """
        # Random moves (tradeoff between exploration and exploitation)
        # Better agent gets the less random we want (more logic-y)
        if self.hiddenToggle == True: 
            self.epsilon = max(1, 180-self.n_games)
        else:
            if self.n_games <= 10:
                self.epsilon = 200
            else:
                self.epsilon = 10
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # Basically every cycle epsilon gets smaller
            # When epsilon is smaller it is less likely for the random thing to generate random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # Basically takes the highest activated neuron in the output layer
            final_move[move] = 1 # This wasn't here
        return final_move

def train():
    """
    Responsible for running functions which trains the model, as well as sending/logging info to graph.
    Args: None
    Returns: None
    """
    # Define variables that are only used in thie function
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    numLeft = 0
    numRight = 0
    numTUp = 0
    numTDown = 0
    numShoot = 0
    numNothing = 0
    agent = Agent()
    game = Game()
    running = True
    while running:
        # Get old state
        state_old = agent.get_state(game)
        
        # get move for statistics
        final_move = agent.get_action(state_old)
        if np.array_equal(final_move, [1, 0, 0]):
            numLeft += 1
        if np.array_equal(final_move, [0, 1, 0]):
            numRight += 1
        if np.array_equal(final_move, [0, 0, 1, 0, 0, 0]):
            numTUp += 1
        if np.array_equal(final_move, [0, 0, 0, 1, 0, 0] ):
            numTDown += 1
        if np.array_equal(final_move, [0, 0, 0, 0, 1, 0] ):
            numShoot += 1
        if np.array_equal(final_move, [0, 0, 1] ):
            numNothing += 1
            
        # perform move and get new state
        reward, done, score = game.playStep(final_move)
        state_new = agent.get_state(game)
        
        # Train short mem
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done: #When a game ends
            # train long mem, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            # Saves model if the AI gets a new record
            if score > record:
                record = score
                agent.model.save(f"game_{agent.n_games}_score_{score}")
                print(f"New record! Saved model at game {agent.n_games}")

                
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # If player toggled to show graphs show graphs. I made this bcuz the graphs popping up was annoying.
            if game.graphToggle:
                plot(plot_scores, plot_mean_scores, numLeft, numRight, numTUp, numTDown, numShoot, numNothing)
#             plot(numLeft, numRight, numTUp, numTDown, numShoot, numNothing)
            
if __name__ == "__main__":
    train()