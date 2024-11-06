from game_lib.game_utilities.helper_functions import *
import time
import math
import matplotlib.pyplot as plt
from ml_lib.multiagentsreinforcmentlearning_model import  *

class SnakeGame:

    def __init__(self, w=600, h=600, num_snakes=1):
        self.w = w
        self.h = h
        self.background_image = pygame.transform.scale(pygame.image.load("game_lib/game_images/background_image.jpg"), (self.w, self.h))
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.num_snakes = num_snakes
        self.snakes = []
        self.frames_since_last_action = [0] * self.num_snakes
        self.MAX_FRAMES_INACTIVITY = 100000
        self.start_time = []

        self.EatenTeamMate = [0] * self.num_snakes
        self.EatenOponent = [0] * self.num_snakes
        self.AppleSameColor = [0] * self.num_snakes
        self.GiftedApples = [0] * self.num_snakes
        self.OponentApples = [0] * self.num_snakes
        self.TeamSCore = [0] * NumberOFTeams

        self.colors = [
            "white",
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
            "orange",
        ]

        self.RestrictedColors = ["green",
                                 "blue"]

        # ini tiziqlizate the colors of the snales
        current_index = 0
        self.SnakeColors = []  # the snakes  colors
        for _ in range(self.num_snakes):
            # RandomFoodIndex
            current_index = current_index % 2
            # choose  only two colors
            self.SnakeColors.append(self.colors[current_index])
            current_index += 1
        self.reset()

        # it can happedb that the length of food it not the same length of the snakes
        self.FoodColors = []
        current_index = 0
        for index in range(NUM_APPLES):
            # RandomFoodIndex

            current_index = current_index % 4

            self.FoodColors.append(self.colors[current_index])
            current_index += 1

        self.reset()

    def reset_snake(self, snake_index):

        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        # food  color

        self.heads[snake_index] = Point(x, y)
        self.snakes[snake_index] = [self.heads[snake_index],
                                    Point(self.heads[snake_index].x - BLOCK_SIZE, self.heads[snake_index].y),
                                    Point(self.heads[snake_index].x - (2 * BLOCK_SIZE), self.heads[snake_index].y)]
        self.directions[snake_index] = Direction.RIGHT
        self.game_over[snake_index] = False
        self.score[snake_index] = 0
        self.start_time[snake_index] = time.time()
        self.Apple_EatenSnakes[snake_index] = [0, 0]

    def reset(self):
        self.frame_iteration = 0
        for _ in range(self.num_snakes):
            self.start_time.append(time.time())
        self.score = [0] * (self.num_snakes)

        self.Apple_EatenSnakes = [[0, 0]] * (self.num_snakes)
        self.directions = [Direction.RIGHT for _ in range(self.num_snakes)]
        self.heads = []  # List to store snake head positions
        self.snakes = []  # List to store snake body segments
        self.game_over = [False] * self.num_snakes
        # Generate random starting positions for each snake
        for _ in range(self.num_snakes):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            head = Point(x, y)
            self.heads.append(head)
            snake = [head, Point(head.x - BLOCK_SIZE, head.y), Point(head.x - (2 * BLOCK_SIZE), head.y)]
            self.snakes.append(snake)

        self.food = []  # List to store apple positions
        self._place_food()  # Generate initial apples

    def convert_colors_to_binary(self, color_index):

        binary_colors = []
        if color_index == 0:
            binary_colors.append([0, 0, 0])
        elif color_index == 1:
            binary_colors.append([0, 0, 1])
        elif color_index == 2:
            binary_colors.append([0, 1, 0])
        elif color_index == 3:
            binary_colors.append([0, 1, 1])
        elif color_index == 4:
            binary_colors.append([1, 0, 0])
        elif color_index == 5:
            binary_colors.append([1, 0, 1])
        elif color_index == 6:
            binary_colors.append([1, 1, 0])
        elif color_index == 7:
            binary_colors.append([1, 1, 1])
        else:
            print("Invalid color index.")

        return binary_colors

    def get_state(self):

        states = []

        for snake_index in range(self.num_snakes):

            # Access the head of the snake
            head = self.heads[snake_index]
            # All the pixels in the vicinity of the snake

            point_l = Point(head.x - BLOCK_SIZE, head.y)
            point_r = Point(head.x + BLOCK_SIZE, head.y)
            point_u = Point(head.x, head.y - BLOCK_SIZE)
            point_d = Point(head.x, head.y + BLOCK_SIZE)

            # Check which direction our snake has taken

            dir_l = self.directions[snake_index] == Direction.LEFT
            dir_r = self.directions[snake_index] == Direction.RIGHT
            dir_u = self.directions[snake_index] == Direction.UP
            dir_d = self.directions[snake_index] == Direction.DOWN

            # Collect positions, lengths, and actions of other snakes, color of others snakes

            opponent_positions = []
            opponent_lengths = []
            opponent_actions = []
            oponent_colors = []

            for snake_idx in range(self.num_snakes):

                if snake_idx != snake_index:

                    opponent_positions.append(self.heads[snake_idx])
                    opponent_lengths.append(len(self.snakes[snake_idx]))  # Append the length of each opponent snake

                    # get the oponent action
                    opponent_action = [0, 0, 0, 0]
                    if self.directions[snake_idx] == Direction.LEFT:
                        opponent_action = [1, 0, 0, 0]
                    elif self.directions[snake_idx] == Direction.RIGHT:
                        opponent_action = [0, 1, 0, 0]
                    elif self.directions[snake_idx] == Direction.UP:
                        opponent_action = [0, 0, 1, 0]
                    elif self.directions[snake_idx] == Direction.DOWN:
                        opponent_action = [0, 0, 0, 1]
                    opponent_actions.append(opponent_action)

            state = [

                int((dir_r and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_d)))),
                # Danger right
                int((dir_u and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_d)))),
                # Danger left
                int((dir_d and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_d)))),

                # Move direction

                int(dir_l),
                int(dir_r),
                int(dir_u),
                int(dir_d)

            ]

            # add the oponent move

            for action in opponent_actions:
                state += action

                # add  lengths
            my_length = len(self.snakes[snake_index])

            for opponent_length in opponent_lengths:

                if my_length > opponent_length:
                    state += [1, 0]  # Snake length is greater than opponent
                elif my_length < opponent_length:
                    state += [0, 1]  # Snake length is smaller than opponent
                else:
                    state += [0, 0]  # Snake length is equal to opponent

            # add the food location to each snakes

            for food_item in self.food:
                state += [
                    int(food_item.x < head.x),  # food left
                    int(food_item.x > head.x),  # food right
                    int(food_item.y < head.y),  # food up
                    int(food_item.y > head.y)  # food down
                ]

            # we nnedd to add the agent colors also

            for color_index in range(len(self.snakes)):

                item = self.SnakeColors[color_index]

                value = self.convert_colors_to_binary(self.colors.index(item))  # Food color

                for item in value:
                    state.extend(item)

                    # add the food colors

            for food in range(len(self.food)):

                # get the color name  of first item in the food

                item = self.FoodColors[food]

                value = self.convert_colors_to_binary(self.colors.index(item))  # Food color

                for item in value:
                    state.extend(item)

            states.append(np.array(state, dtype=int))

        return states

    def CheckForGetState(self, snake_index, pt=None):

        if pt is None:
            pt = self.heads[snake_index]

        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Hits itself

        for i, body in enumerate(self.snakes[snake_index][1:]):
            if pt == body:
                return True

        for i, snake in enumerate(self.snakes):
            if i != snake_index:
                if pt in snake:
                    if self.SnakeColors[i] == self.SnakeColors[snake_index] or len(snake) >= len(
                            self.snakes[snake_index]):
                        return True

        if pt in self.food:
            for i, apple in enumerate(self.food):
                # print(f" the currnet apple colro  : {self.FoodColors[i]} , the current snake  :{self.SnakeColors[snake_index]} ")

                if pt == apple and self.FoodColors[i] in self.RestrictedColors:
                    return True

        return False

    def _place_food(self):

        self.food = []
        for _ in range(NUM_APPLES):  # NUM_APPLES is the desired number of apples
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food.append(Point(x, y))

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def play_step(self, actions):

        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        snake_info = ["none"] * self.num_snakes
        total_time_played = [[] for _ in range(self.num_snakes)]
        rewards = [0] * self.num_snakes

        # take actions
        self._move(actions)

        # calculate the ditance to each apple and each snake
        AppleDistanceList = []
        DistanceToSnakesList = []

        for snake_index in range(self.num_snakes):
            AppleDistance = []
            DistanceToSnakes = []

            for i, food in enumerate(self.food):
                apple_distance = self.calculate_distance(self.snakes[snake_index][0], self.food[i])
                AppleDistance.append(apple_distance)

            AppleDistanceList.append(AppleDistance)

            for i, snake in enumerate(self.snakes):
                if i != snake_index:
                    snake_distance = self.calculate_distance(self.snakes[snake_index][0], self.snakes[i][0])
                    DistanceToSnakes.append(snake_distance)

            DistanceToSnakesList.append(DistanceToSnakes)

            # elapsed time

            elapsed_time = time.time() - self.start_time[snake_index]
            # get head  of the current snake
            snake_head = self.heads[snake_index]

            # check collison with the wall

            if self.collsion_wall(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = -50
                self.frames_since_last_action[snake_index] = 0
                # self.reset_snake(snake_index)

                snake_info[snake_index] = "I collided with the wall"

                # check collison with itself
            elif self.collison_with_itself(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = -60
                self.frames_since_last_action[snake_index] = 0
                # self.reset_snake(snake_index)

                snake_info[snake_index] = "I collided with myself"

                # other wise
            else:
                self.snakes[snake_index].insert(0, snake_head)
                # check if we collided with the food
                if snake_head in self.food:
                    # print(f" the current self.food lsit : {self.food}")

                    # increment the number of eaten apples
                    self.Apple_EatenSnakes[snake_index][1] += 1

                    # find out wich apple we ate
                    for i, apple in enumerate(self.food):
                        # print(f"the current i is : {i} the color of apple  : {self.FoodColors[i] } , collided with snake color: {self.SnakeColors[snake_index]}")

                        if snake_head == apple:

                            # 1- step  : if  we ate gifted  apple
                            if self.FoodColors[i] in self.RestrictedColors:

                                total_time_played[snake_index] = elapsed_time
                                self.game_over[snake_index] = True
                                rewards[snake_index] = -70
                                # self.reset_snake(snake_index)

                                # print(f" i eated a gifted apple !! ")

                                snake_info[snake_index] = "I ate gifted apple!"
                                self.GiftedApples[snake_index] += 1

                                # step 2  : if  both snake and apple have the same  color , else check the other condition

                            else:
                                if self.SnakeColors[snake_index] == self.FoodColors[i]:
                                    self.score[snake_index] += 1
                                    rewards[snake_index] = 90
                                    # print(f"I ate  apple wich have same color as me ")
                                    self.AppleSameColor[snake_index] += 1
                                    snake_info[snake_index] = "I ate  apple wich have same color as me !"

                                else:
                                    # print(f"I ate just normal apple!")
                                    self.score[snake_index] += 1
                                    rewards[snake_index] = 70
                                    self.OponentApples[snake_index] += 1
                                    snake_info[snake_index] = "I ate just normal apple!"

                        self.food.pop(i)  # Remove the eaten apple
                        self._place_food()  # Generate a new apple to replace the eaten one

                    self.frames_since_last_action[snake_index] = 0

                else:
                    self.snakes[snake_index].pop()
                    self.frames_since_last_action[snake_index] += 1
                    snake_info[snake_index] = "exploring the env !"

                    for other_snake_index in range(self.num_snakes):
                        # check  if the current snake  is  not  our  snake
                        if snake_index != other_snake_index:

                            # check for collision

                            if snake_head in self.snakes[other_snake_index]:

                                # check if we collided with a our team mate
                                # print(f"the current snake color  : {self.SnakeColors[snake_index]} , oponent : {self.SnakeColors[other_snake_index]}")
                                if self.SnakeColors[snake_index] == self.SnakeColors[other_snake_index]:
                                    rewards[snake_index] = -80  # Encourage collision with longer snakes
                                    snake_info[snake_index] = "collided with my team mate !"

                                    if self.SnakeColors[snake_index] == "white":

                                        self.TeamSCore[0] -= 1

                                    else:

                                        self.TeamSCore[1] -= 1

                                    self.EatenTeamMate[snake_index] += 1
                                    self.game_over[snake_index] = True
                                    # self.reset_snake(snake_index)


                                else:

                                    if len(self.snakes[snake_index]) <= len(self.snakes[other_snake_index]):
                                        rewards[snake_index] = -100  # Encourage collision with longer snakes
                                        snake_info[snake_index] = "oponent snake ate me!"
                                        self.game_over[snake_index] = True
                                        # self.reset_snake(snake_index)


                                    elif len(self.snakes[snake_index]) > len(self.snakes[other_snake_index]):
                                        rewards[snake_index] = 100  # Encourage collision with shorter snakes
                                        snake_info[snake_index] = "I ate  oponent snake!"
                                        if self.SnakeColors[snake_index] == "white":
                                            self.TeamSCore[0] += 1
                                        else:
                                            self.TeamSCore[1] += 1
                                        self.EatenOponent[snake_index] += 1
                                        self.score[snake_index] += 1

                    total_time_played[snake_index] = elapsed_time

                    if self.frames_since_last_action[snake_index] >= self.MAX_FRAMES_INACTIVITY:
                        self.game_over[snake_index] = True
                        rewards[snake_index] = -10
                        # self.reset_snake(snake_index)
                        self.frames_since_last_action[snake_index] = 0
                        # self.reset_snake(snake_index)

                        snake_info[snake_index] = "I didn't do anything for n iterations"

                        # how to provide total  number of  eatne gifted snake

        # print(f" the snakes  colros are   : {self.SnakeColors}")

        return rewards, self.game_over, self.score, snake_info, total_time_played, self.Apple_EatenSnakes, DistanceToSnakesList, AppleDistanceList

    def is_collision(self, snake_index, pt=None):

        if pt is None:
            pt = self.heads[snake_index]

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            # hits itself
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True
        return False

    def collsion_wall(self, snake_index, pt=None):
        if pt is None:
            pt = self.heads[snake_index]
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False

    def collison_with_itself(self, snake_index, pt=None):
        if pt is None:
            pt = self.heads[snake_index]
        # hits itself

        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True

    def eate_other_snake(self, snake_index):

        head = self.heads[snake_index]
        truth = False
        collided = []
        for i, snake in enumerate(self.snakes):
            if i != snake_index and head in snake:
                # get the index of the hitted snake
                collided.append(i)
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])
                if current_snake_length > collided_snake_length:
                    truth = True
        return truth, collided

    def eaten_by_other_snake(self, snake_index):
        head = self.heads[snake_index]
        for i, snake in enumerate(self.snakes):
            if i != snake_index and head in snake:
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])
                if current_snake_length <= collided_snake_length:
                    return True
        return False

    def grid(self):
        for row in range(0, self.h, BLOCK_SIZE):
            for col in range(0, self.h, BLOCK_SIZE):
                # draw rect
                rect = pygame.Rect(row, col, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, "green", rect, 3)
        pygame.display.update()

    def _update_ui(self):

        self.display.fill((0, 0, 0))
        self.display.blit(self.background_image, (0, 0))

        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Green', (x, 0), (x, self.h), 2)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Green', (0, y), (self.w, y), 2)

        for snake_index in range(self.num_snakes):
            for point in self.snakes[snake_index]:
                pygame.draw.rect(
                    self.display,
                    self.SnakeColors[snake_index],
                    pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE)
                )

            for i, food in enumerate(self.food):
                # choose food colors

                pygame.draw.rect(
                    self.display,
                    self.FoodColors[i],
                    pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE)
                )
            '''
            if self.game_over[snake_index]:
                font = pygame.font.Font(None, 50)
                text = font.render("Game Over", True, (255, 255, 255))
                self.display.blit(text, (self.w // 2 - text.get_width() // 2, self.h // 2 - text.get_height() // 2))
            '''

            score_font = pygame.font.Font(None, 36)
            score_text = score_font.render("Score: " + str(self.score[snake_index]), True, (255, 255, 255))
            self.display.blit(score_text, (10, 10 + 40 * snake_index))

        pygame.display.flip()
        self.clock.tick(FPS)

    def handle_user_input(self):

        # the action taken by the  human agent

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return [0, 0, 1]  # Left turn action
        elif keys[pygame.K_DOWN]:
            return [0, 1, 0]  # Right turn action
        else:
            return [1, 0, 0]  # No change action

    def _move(self, actions):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        for snake_index in range(self.num_snakes):

            idx = clock_wise.index(self.directions[snake_index])
            if np.array_equal(actions[snake_index], [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(actions[snake_index], [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
            self.directions[snake_index] = new_dir
            x = self.heads[snake_index].x
            y = self.heads[snake_index].y
            if self.directions[snake_index] == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.directions[snake_index] == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.UP:
                y -= BLOCK_SIZE
            self.heads[snake_index] = Point(x, y)


def Create_agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma):
    return Agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma)


def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)