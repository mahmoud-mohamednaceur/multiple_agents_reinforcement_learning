import numpy as np

class ReplayBuffer:
    """
    A class that implements a replay buffer to store transitions
    for experience replay in reinforcement learning.

    Attributes:
        capacity (int): Maximum number of transitions the buffer can hold.
        input_dim (int): Dimensionality of the input state.
        n_actions (int): Number of possible actions.
        mem_cntr (int): Current number of stored transitions.
        states (numpy.ndarray): Array to store current states.
        next_states (numpy.ndarray): Array to store next states.
        actions (numpy.ndarray): Array to store actions taken.
        rewards (numpy.ndarray): Array to store rewards received.
        dones (numpy.ndarray): Array to store done flags for each transition.
    """

    def __init__(self, capacity, input_dim, n_actions):
        """
        Initializes the replay buffer with specified capacity, input dimensions,
        and number of actions. Sets up storage for states, actions, rewards,
        and done flags.

        Parameters:
            capacity (int): Maximum number of transitions to store in the buffer.
            input_dim (int): Dimensionality of the input state.
            n_actions (int): Number of possible actions.
        """
        self.capacity = capacity
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.mem_cntr = 0  # Tracks the current number of transitions stored
        self.states = np.zeros((self.capacity, self.input_dim))  # Current states
        self.next_states = np.zeros((self.capacity, self.input_dim))  # Next states
        self.actions = np.zeros((self.capacity, self.n_actions))  # Actions taken
        self.rewards = np.zeros(self.capacity)  # Rewards received
        self.dones = np.zeros(self.capacity, dtype=bool)  # Done flags for each transition

    def store_transition(self, state, next_state, action, reward, done):
        """
        Stores a transition (state, action, reward, next_state, done) in the buffer.
        If the buffer is full, it overwrites the oldest transition.

        Parameters:
            state (numpy.ndarray): The current state.
            next_state (numpy.ndarray): The next state after taking the action.
            action (numpy.ndarray): The action taken.
            reward (float): The reward received from taking the action.
            done (bool): A flag indicating if the episode has ended.
        """
        index = self.mem_cntr % self.capacity  # Calculate the index to store the transition
        self.states[index] = state  # Store the current state
        self.next_states[index] = next_state  # Store the next state
        self.actions[index] = action  # Store the action taken
        self.rewards[index] = reward  # Store the reward received
        self.dones[index] = done  # Store the done flag
        self.mem_cntr += 1  # Increment the transition counter

    def sample_batch(self, batch_size):
        """
        Samples a random batch of transitions from the replay buffer.

        Parameters:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A tuple containing arrays of states, next states, actions,
                   rewards, and done flags for the sampled transitions.
        """
        max_mem = min(self.mem_cntr, self.capacity)  # Maximum transitions to sample from

        # Ensure that we sample without replacement if we have enough memory
        if max_mem >= batch_size:
            batch_indices = np.random.choice(max_mem, batch_size, replace=False)
        else:
            batch_indices = np.random.choice(max_mem, batch_size, replace=True)

        # Retrieve the sampled transitions
        states = self.states[batch_indices]
        next_states = self.next_states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        dones = self.dones[batch_indices]

        return states, next_states, actions, rewards, dones
