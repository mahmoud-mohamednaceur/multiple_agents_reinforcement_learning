from enum import Enum  # Import Enum for defining constant values
from collections import namedtuple  # Import namedtuple for creating simple classes
import pygame  # Import the pygame library for game development
import matplotlib.pyplot as plt  # Import matplotlib for plotting graphs
from ml_lib.multiagentsreinforcmentlearning_model import  *
# Initialize the pygame library
pygame.init()

# Set the font for rendering text
font = pygame.font.SysFont("comicsans", 50)

# Enum class to represent possible movement directions
class Direction(Enum):
    RIGHT = 1  # Move to the right
    LEFT = 2   # Move to the left
    UP = 3     # Move upwards
    DOWN = 4   # Move downwards

# Named tuple to represent a point in 2D space (x, y coordinates)
Point = namedtuple('Point', 'x, y')

# Color definitions using RGB tuples
WHITE = (255, 255, 255)   # White color
RED = (200, 0, 0)          # Red color
BLUE1 = (0, 0, 255)        # Blue color (bright)
BLUE2 = (0, 100, 255)      # Blue color (darker)
BLACK = (0, 0, 0)          # Black color

# Game configuration constants
BLOCK_SIZE = 30            # Size of each block in the game grid
SPEED = 50                 # Speed of the game (in some units, e.g., pixels per second)
FPS = 50                    # Frames per second for the game loop
NUM_APPLES = 5             # Number of apples to be generated in the game
NumberOFTeams = 2          # Number of teams involved in the game

def Create_agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma):
    """
    Factory function to create an Agent instance with the specified parameters.

    Parameters:
        input_dim (int): Input dimension for the agent's neural network.
        dim1 (int): Dimension for the first layer of the neural network.
        dim2 (int): Dimension for the second layer of the neural network.
        n_actions (int): Number of possible actions for the agent.
        lr (float): Learning rate for the agent's optimizer.
        butch_size (int): Batch size for learning.
        mem_size (int): Size of the memory buffer for storing experiences.
        gamma (float): Discount factor for future rewards.

    Returns:
        Agent: An instance of the Agent class initialized with the provided parameters.
    """
    return Agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma)

def plot(scores, mean_scores):
    """
    Plots the training scores and mean scores over the number of games played.

    Parameters:
        scores (list): List of scores obtained in each game.
        mean_scores (list): List of mean scores calculated over a sliding window.

    This function updates the plot to display the current scores and mean scores.
    It also sets the title and labels for the axes.
    """
    plt.clf()  # Clear the current figure
    plt.title('Training...')  # Set the title of the plot
    plt.xlabel('Number of Games')  # Label for the x-axis
    plt.ylabel('Score')  # Label for the y-axis
    plt.plot(scores)  # Plot the scores obtained in each game
    plt.plot(mean_scores)  # Plot the mean scores
    plt.ylim(ymin=0)  # Set the minimum limit for the y-axis to 0
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # Display the latest score
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))  # Display the latest mean score
    plt.show(block=False)  # Display the plot without blocking the execution
    plt.pause(.1)  # Pause briefly to allow the plot to update
