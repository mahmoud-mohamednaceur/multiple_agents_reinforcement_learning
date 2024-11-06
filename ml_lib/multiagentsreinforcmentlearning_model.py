import torch
from ml_lib.ml_utilities.replaybuffermultiagentsreinforcementlearning import *
from ml_lib.ml_utilities.multiagentsneuralnetworks import Qnetwork, SimpleNetwork, SimpleDiffrentLossFunction, \
    SimpleNetworkWithDiffrentOptimizer, MoreLayerDiffrentLossFunction, MoreLayersNetwork, \
    MoreLayersNetworkDiffrentOptimizer
import random
import os


class Agent:
    """
    The Agent class manages multiple reinforcement learning agents, each with its own neural network,
    replay buffer, and hyperparameters.

    Attributes:
        num_agents (int): The number of agents to manage.
        evaluate (bool): Flag indicating whether the agents are in evaluation mode.
        agents (list): List of dictionaries holding agent-specific information.
        gamma_list (list): List of discount factors (gamma) for each agent.
        batch_size (int): The batch size for learning.
    """

    def __init__(self, input_dimlsit, fc1_dimlsit, fc2_dimlist, fc3_dimlist, fc4_dimlist, n_actions, lrlist, losslist,
                 batch_size, mem_size, gamma_list, num_agents):
        """
        Initializes the Agent class with specified parameters for each agent.

        Parameters:
            input_dimlsit (list): List of input dimensions for each agent.
            fc1_dimlsit (list): List of first layer dimensions for each agent.
            fc2_dimlist (list): List of second layer dimensions for each agent.
            fc3_dimlist (list): List of third layer dimensions for each agent.
            fc4_dimlist (list): List of fourth layer dimensions for each agent.
            n_actions (int): Number of possible actions for each agent.
            lrlist (list): List of learning rates for each agent.
            losslist (list): List of loss functions for each agent.
            batch_size (int): Batch size used for learning.
            mem_size (int): Capacity of the replay buffer for each agent.
            gamma_list (list): List of discount factors (gamma) for each agent.
            num_agents (int): Total number of agents.
        """
        self.num_agents = num_agents  # Store the number of agents
        self.evaluate = False  # Evaluation flag for the agents
        self.agents = []  # List to hold each agent's information
        self.gamma_list = gamma_list  # List of gamma values for each agent
        self.batch_size = batch_size  # Set the batch size for learning

        # List of available network architectures
        Networks_list = [Qnetwork, MoreLayerDiffrentLossFunction, SimpleNetworkWithDiffrentOptimizer,
                         MoreLayerDiffrentLossFunction, MoreLayersNetwork, MoreLayersNetworkDiffrentOptimizer]

        # Initialize each agent with its own network and replay buffer
        for index in range(num_agents):
            input_dim = input_dimlsit[index]
            fc1_dim = fc1_dimlsit[index]
            fc2_dim = fc2_dimlist[index]
            fc3_dim = fc3_dimlist[index]
            fc4_dim = fc4_dimlist[index]
            lr = lrlist[index]
            loss = losslist[index]

            # Create a replay buffer for the agent
            agent_mem = ReplayBuffer(mem_size, input_dim, n_actions)
            # Every agent will have its own network
            agent_network = Qnetwork(input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_actions, lr, loss)
            gamma = gamma_list[index]  # Assign the gamma value for this agent

            # Create a dictionary to hold agent information
            agent = {
                'mem': agent_mem,
                'network': agent_network,
                'epsilon': 0,  # Exploration parameter
                'n_games': 0,  # Number of games played
                'gamma': gamma  # Assign gamma value to agent
            }

            self.agents.append(agent)  # Add the agent to the list

    def choose_action(self, states):
        """
        Chooses an action for each agent based on the current state using an epsilon-greedy strategy.

        Parameters:
            states (list): List of current states for each agent.

        Returns:
            list: List of actions selected for each agent.
        """
        actions = []  # List to hold actions for each agent

        for agent_index, agent in enumerate(self.agents):
            # Calculate epsilon for exploration-exploitation trade-off
            epsilon = 1000 - agent['n_games']
            final_move = [0, 0, 0]  # Initialize action array

            # Epsilon-greedy action selection
            if random.randint(0, 1000) < epsilon:
                move = random.randint(0, 2)  # Explore: select a random action
                final_move[move] = 1
            else:
                state = states[agent_index]
                state_tensor = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
                prediction = agent['network'](state_tensor)  # Get action predictions from the network
                move = torch.argmax(prediction).item()  # Select the action with the highest predicted value
                final_move[move] = 1  # Update the final action array

            actions.append(final_move)  # Add action to the list
        return actions  # Return actions for all agents

    def short_mem(self, states, next_states, actions, rewards, dones):
        """
        Stores transitions in the replay buffer for each agent and triggers the learning process.

        Parameters:
            states (list): List of current states for each agent.
            next_states (list): List of next states for each agent.
            actions (list): List of actions taken by each agent.
            rewards (list): List of rewards received by each agent.
            dones (list): List of done flags for each agent.
        """
        for agent_index, agent in enumerate(self.agents):
            # Store transitions in the replay buffer for each agent
            agent['mem'].store_transition(states[agent_index], next_states[agent_index],
                                          actions[agent_index], rewards[agent_index], dones[agent_index])
            agent['n_games'] += 1  # Increment number of games played
            agent['epsilon'] = 100 - agent['n_games']  # Update epsilon

        self.learn()  # Trigger learning process after storing transitions

    def long_mem(self):
        """
        Checks if any agent has enough memory to learn from and triggers learning.
        """
        # Check if any agent has enough memory to learn from
        for agent in self.agents:
            if self.batch_size < agent['mem'].mem_cntr:  # Ensure enough samples are available
                self.learn()  # Trigger learning process

    def long_memory(self, AgentIndex):
        """
        Checks memory for a specific agent and triggers learning if enough samples are available.

        Parameters:
            AgentIndex (int): Index of the agent to check memory for.
        """
        # Check memory for a specific agent
        agent = self.agents[AgentIndex]
        if self.batch_size < agent['mem'].mem_cntr:  # Ensure enough samples are available
            self.learn()  # Trigger learning process

    def save(self, agent_idx, color, Zeitpunkt):
        """
        Saves the model parameters of a specific agent.

        Parameters:
            agent_idx (int): Index of the agent to save.
            color (str): Color identifier for the model file.
            Zeitpunkt (str): Timestamp for the model save.
        """
        file_name = f'agent{agent_idx}model_saved_after{Zeitpunkt}_iterations.pth'
        model_folder_path = f'output_folder/trained_models/agent_{agent_idx}'  # Define model save path
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)  # Create directory if it doesn't exist
        agent = self.agents[agent_idx]  # Get specific agent
        file_name = os.path.join(model_folder_path, file_name)
        file_name_agent = f'{file_name}_agent_{agent_idx}'
        torch.save(agent['network'].state_dict(), file_name_agent)  # Save network state

    def load_model(self, saved_model_path):
        """
        Loads the saved model parameters for a specific agent.

        Parameters:
            saved_model_path (str): Path to the saved model file.

        Returns:
            model (Qnetwork): The model with loaded parameters.
        """
        # Load the saved model parameters
        saved_model = torch.load(saved_model_path)  # Load model state dictionary
        model = Qnetwork(self.input_dim, self.fc1_dim, self.fc2_dim, self.fc3_dim, self.fc4_dim, self.n_actions,
                         self.lr, self.loss)
        model.load_state_dict(saved_model)  # Load parameters into the model

        return model  # Return the model with loaded parameters

    def loadmodel(self, saved_path_list):
        """
        Loads models for all agents from a list of saved paths.

        Parameters:
            saved_path_list (list): List of paths for saved agent models.

        Returns:
            list: List of models for each agent.
        """
        model_list = []  # Initialize model list
        for index in range(self.num_agents):
            model = self.load_model(saved_path_list[index])  # Load each agent's model
            model_list.append(model)  # Add to model list
        return model_list  # Return the list of loaded models

    def learn(self):
        """
        Performs the learning step for each agent by updating the network parameters based on sampled transitions.
        """
        # Learning step for each agent
        for agent_index, agent in enumerate(self.agents):
            states, next_states, actions, rewards, dones = agent['mem'].sample_batch(
                self.batch_size)  # Sample a batch from memory

            state_tensor = torch.tensor(states, dtype=torch.float)  # Convert states to tensor
            next_state_tensor = torch.tensor(next_states, dtype=torch.float)  # Convert next states to tensor
            action_tensor = torch.tensor(actions, dtype=torch.long)  # Convert actions to tensor
            reward_tensor = torch.tensor(rewards, dtype=torch.float)  # Convert rewards to tensor
            done_tensor = torch.tensor(dones)  # Convert dones to tensor

            # Ensure tensors are of shape (batch_size, ...)
            if len(state_tensor.shape) == 1:
                state_tensor = torch.unsqueeze(state_tensor, 0)
                next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
                action_tensor = torch.unsqueeze(action_tensor, 0)
                reward_tensor = torch.unsqueeze(reward_tensor, 0)
                done_tensor = torch.unsqueeze(done_tensor, 0)

            pred = agent['network'](state_tensor)  # Get predictions from the agent's network

            target = pred.clone()  # Clone predictions for target calculations
            for idx in range(len(done_tensor)):
                Q_new = reward_tensor[idx]  # Initialize Q value with reward
                if not done_tensor[idx]:  # If the episode is not done
                    Q_new = reward_tensor[idx] + agent['gamma'] * torch.max(
                        agent['network'](next_state_tensor[idx]))  # Update Q value

                target[idx][
                    torch.argmax(action_tensor[idx]).item()] = Q_new  # Update the target for the specific action taken

            agent['network'].optimizer.zero_grad()  # Clear gradients
            loss = agent['network'].loss(target, pred)  # Calculate loss
            loss.backward()  # Backpropagation
            agent['network'].optimizer.step()  # Update network parameters
