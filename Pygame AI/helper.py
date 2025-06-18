"""
Summative - AI in PyGame
ICS3U-03
Vincent Pham
Helper file (plots using matplotlib)
History: Version 1 - May 30, 2025
"""

# BASICS - Imports
import matplotlib.pyplot as plt
from IPython import display

# Activating the library
plt.ion()

def plot(scores, mean_scores, numLeft, numRight, numTUp, numTDown, numShoot, numNothing):
    """
    Plots the two graphs when the game ends
    Args:
    scores (list) - list of past scores
    mean_score (list) - list of mean scores that are recalculated every game
    numLeft (int) - amount of times AI chose to turn left
    numRight (int) - amount of time AI chose to turn right
    numTUp (int) - amount of times AI chose to choose Throttle Up
    numTDown (int) - amount of times AI chose to Throttle Down
    numShoot (int) - amount of times AI chose to Shoot
    numNothing (int) - amount of times AI chose to do nothing
    """
    # Deletes all other graphs (usually should be one)
    display.clear_output(wait=True)
    plt.close('all')
    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # First subplot - Scores
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    # Second subplot - Actions
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    action_names = ['Left', 'Right', 'TUp', 'TDown', 'Shoot', 'Nothing']
    action_counts = [numLeft, numRight, numTUp, numTDown, numShoot, numNothing]
    plt.bar(action_names, action_counts)
    plt.title('Actions Taken')
    plt.xlabel('Action Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels on top of each bar
    for i, count in enumerate(action_counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    
    plt.tight_layout()  # Prevent overlapping
    display.display(plt.gcf())
    plt.show(block=False)
    plt.pause(.1)