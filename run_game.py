# Import necessary libraries
import torch.nn as nn
import pandas as pd
from game_lib.game_core_functions import *  # Custom library for game functions

# Main execution block
if __name__ == '__main__':

    # Set up game parameters
    number_of_snakes = 2  # Number of snake agents in the game
    current_max = 0  # Track the maximum score achieved

    # Initialize the Snake game with specified number of snakes
    game = SnakeGame(num_snakes=number_of_snakes)

    # Initialize agent with neural network configuration and training parameters
    agent = Agent(input_dimlsit=[54, 54], fc1_dimlsit=[400, 300], fc2_dimlist=[512, 512],
                  fc3_dimlist=[256, 256], fc4_dimlist=[256, 256], n_actions=3,
                  losslist=[nn.MSELoss(), nn.MSELoss()], lrlist=[0.004, 0.002],
                  batch_size=10, mem_size=10000, gamma_list=[0.30, 0.40], num_agents=2)

    # Game variables to store statistics and track game state
    running = True
    step = [0] * game.num_snakes  # Track rounds per snake agent

    # Cumulative game statistics for each snake
    TotalPlayerScorePro = [0] * game.num_snakes
    Total_PlayedTime = [0] * game.num_snakes
    TotalTimeBeforeDeath = [0] * game.num_snakes
    TotalSnakeEaten = [0] * game.num_snakes
    TotalAppleEaten = [0] * game.num_snakes

    # Nested lists for tracking each snake's performance in detail
    TotalEatenSnakes = [[] for _ in range(game.num_snakes)]
    TotalEatenApples = [[] for _ in range(game.num_snakes)]
    Total_score_list = [[] for _ in range(game.num_snakes)]
    Total_Time_List = [[] for _ in range(game.num_snakes)]

    DataFrames = []  # List to store performance data for each snake
    BestPerformance = [0] * game.num_snakes  # Track best score per snake

    # Initialize data tracking dictionaries for each snake agent
    for agent_idx in range(number_of_snakes):
        data = {
            f'n_games{agent_idx}': [],
            f'playerScoreProRound{agent_idx}': [],
            f'playerTotalScore{agent_idx}': [],
            f'TimePlayedPRoRound{agent_idx}': [],
            f'TotalTimePlayed{agent_idx}': [],
            f'MeanScore{agent_idx}': [],
            f'TimeOverScore{agent_idx}': [],
            f'TotalNumberofDeath{agent_idx}': [],
            f'TotalTimeSpendOverTotalTimeOfDeath{agent_idx}': [],
            f'Epsilon{agent_idx}': [],
            f'GifftedApples{agent_idx}': [],
            f'StealedOponentAppels{agent_idx}': [],
            f'ApplesWithSameColorASmine{agent_idx}': [],
            f'EatenOponent{agent_idx}': [],
            f'EatenTeamMates{agent_idx}': [],
            f'TeamScore{agent_idx}': [],
            f'CurrentState{agent_idx}': [],
        }
        DataFrames.append(data)

    # Main game loop for 10,000 iterations
    i = 0
    while i < 10000: # numbre if iterations

        # Fetch current state and choose action for each agent
        old_states = game.get_state()
        actions = agent.choose_action(old_states)

        # Perform a game step and retrieve updated game variables
        rewards, game_over, scores, info, time_played, apple_snake, DistanceToSnakesList, DistanceToAppleList = game.play_step(
            actions)
        game._update_ui()
        game.clock.tick(SPEED)

        # Update agent's short-term memory with experience from this step
        states_new = game.get_state()
        agent.short_mem(old_states, states_new, actions, rewards, game_over)

        # Handle situation when some agents die but game is not over
        if any(game_over) and not all(game_over):

            # Process statistics for each agent that died this step
            indices = [index for index, value in enumerate(game_over) if value]
            for index in indices:
                step[index] += 1  # Increment game round count
                TotalPlayerScorePro[index] += scores[index]
                Total_PlayedTime[index] += time_played[index]
                TotalSnakeEaten[index] += apple_snake[index][1]
                TotalAppleEaten[index] += apple_snake[index][0]

                # Update DataFrame with statistics for the current game round
                DataFrames[index][f'TeamScore{index}'].append(
                    game.TeamSCore[game.SnakeColors.index(game.SnakeColors[index])])
                DataFrames[index][f'GifftedApples{index}'].append(game.GiftedApples[index])
                DataFrames[index][f'StealedOponentAppels{index}'].append(game.OponentApples[index])
                DataFrames[index][f'ApplesWithSameColorASmine{index}'].append(game.AppleSameColor[index])
                DataFrames[index][f'EatenOponent{index}'].append(game.EatenOponent[index])
                DataFrames[index][f'EatenTeamMates{index}'].append(game.EatenTeamMate[index])
                DataFrames[index][f'n_games{index}'].append(step[index])
                DataFrames[index][f'CurrentState{index}'].append(info[index])
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index] / step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'TimePlayedPRoRound{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalTimePlayed{index}'].append(Total_PlayedTime[index])

                # Track time-to-score ratio and death count
                if TotalPlayerScorePro[index] > 0:
                    DataFrames[index][f'TimeOverScore{index}'].append(
                        Total_PlayedTime[index] / TotalPlayerScorePro[index])
                else:
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])
                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TotalTimeSpendOverTotalTimeOfDeath{index}'].append(
                    Total_PlayedTime[index] / step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])

                # Reset game state for the snake that died and update UI
                game.reset_snake(index)
                game._update_ui()
                game.clock.tick(SPEED)
                agent.long_memory(index)

        # Handle case when all agents have died in this step
        elif all(game_over):
            indices = [index for index, value in enumerate(game_over) if value]
            for index in indices:
                # Similar updates as above when a single agent dies
                step[index] += 1
                TotalPlayerScorePro[index] += scores[index]
                Total_PlayedTime[index] += time_played[index]
                TotalSnakeEaten[index] += apple_snake[index][1]
                TotalAppleEaten[index] += apple_snake[index][0]

                # Update all statistics for each agent
                DataFrames[index][f'TeamScore{index}'].append(
                    game.TeamSCore[game.SnakeColors.index(game.SnakeColors[index])])
                DataFrames[index][f'GifftedApples{index}'].append(game.GiftedApples[index])
                DataFrames[index][f'StealedOponentAppels{index}'].append(game.OponentApples[index])
                DataFrames[index][f'ApplesWithSameColorASmine{index}'].append(game.AppleSameColor[index])
                DataFrames[index][f'EatenOponent{index}'].append(game.EatenOponent[index])
                DataFrames[index][f'EatenTeamMates{index}'].append(game.EatenTeamMate[index])
                DataFrames[index][f'n_games{index}'].append(step[index])
                DataFrames[index][f'CurrentState{index}'].append(info[index])
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index] / step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'TimePlayedPRoRound{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalTimePlayed{index}'].append(Total_PlayedTime[index])

                if TotalPlayerScorePro[index] > 0:
                    DataFrames[index][f'TimeOverScore{index}'].append(
                        Total_PlayedTime[index] / TotalPlayerScorePro[index])
                else:
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])
                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TotalTimeSpendOverTotalTimeOfDeath{index}'].append(
                    Total_PlayedTime[index] / step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])

            # Save model for each agent if performance improves
            for agent_index in range(game.num_snakes):
                if BestPerformance[agent_index] < scores[agent_index]:
                    BestPerformance[agent_index] = scores[agent_index]
                    agent.save(agent_index)

            # Reset game state after all agents die
            game.reset()
            game._update_ui()
            game.clock.tick(SPEED)
            agent.long_mem()

        # Save model periodically every 1000 iterations
        for agent_index in range(game.num_snakes):
            if i % 1000 == 0:
                agent.save(agent_index, game.SnakeColors[agent_index], i)

        # Update iteration count and handle quit events
        print(f"step : {i}")
        i += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()  # End game on loop exit

# Save game data for each snake agent to CSV files
save_path = 'output_folder/excel_files'
for dataFrame_index in range(game.num_snakes):
    df = pd.DataFrame(DataFrames[dataFrame_index])
    file_path = os.path.join(save_path, f'agent{dataFrame_index}_performance_over_epochen.csv')
    df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))
