"""
Summative - AI in PyGame
ICS3U-03
Vincent Pham
Model file (trains and predicts next move)
History: Version 1 - May 30, 2025
"""

# BASICS - Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import datetime
class Linear_QNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, hiddenToggle):
        """
        Initializes the Linear_QNet Class
        Args: 
        input_size (int) - number of nodes for the input layer
        hidden_size (int) - number of nodes for hidden layer(s)
        output_size (int) - number of nodes for output lyer
        hiddenToggle (bool) - if the user chose to have a hidden layer or not
        """
        super().__init__()
        
        # Changes how many layers based on if hidden model is chosen or not
        self.hiddenToggle = hiddenToggle
        if self.hiddenToggle == True:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, output_size)
            print('hidden is on')
        else:
            self.linear1 = nn.Linear(input_size, output_size)
            print('hidden is off')

        
    def forward(self, x):
        """
        One forward pass for the model
        Args: 
        x - some calculus thingy
        Returns: 
        x - some calculus thingy
        """
        if self.hiddenToggle == True:
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
        else:
            x = self.linear1(x)
        return x
    
    #Saving the model
    def save(self, file_name=None):
        """
        Saving the model
        Args: 
        file_name (None) - the name of the file when it is saved
        
        Returns: None
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # Save model using stats from game and timestamp
        file_name = f'model_{file_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    
    def load(self, file_path):
        """
        Loads a model that the user chooses
        Args:
        file_path (str) - the path of the model that the user chose
        
        Returns:
        success (bool) - if the model was correctly loaded or not
        """
        success = False
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            self.eval()
            print(f"Model loaded from {file_path}")
            success = True
        return success

    
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        """
        Initializes the QTrainer (defining variables and optimizers/criterions)
        Args:
        model (class) - the Linear Q Net model
        lr (float) - learning rate of the AI (how much the weights are adjusted)
        gamma (float) - how much each training has an impact on it
        
        Returns: None
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        """
        One frame of training
        Args: 
        state (np array) - state of the game
        action (List) - what the AI did
        reward (Int) - The reward that the AI got for the action (in game.py reward is calculated)
        next_state (np array) - state of the game after the action 
        done (bool) - if the game is done or not (If True, it's done)
        Returns: None
        """
        # Converts everything into tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        #1 predited Q values with curent state
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            action_idx = torch.argmax(action)
            if action_idx >= 6:  # Should never happen now, but just in case
                print(f"Invalid action: {action}, replacing with random")
                action_idx = torch.randint(0, 6, (1,))  # Random valid action
        #2 Q_new = r + y * max(next_predicted q value) ->only if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) #QNew and Q
        loss.backward()
        
        self.optimizer.step()