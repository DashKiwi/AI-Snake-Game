import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameLogic, SnakeGameVisual, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.001

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


class VectorizedAgent:
    def __init__(self, num_envs=32):
        self.num_envs = num_envs
        self.agent = Agent()
        self.games = [SnakeGameLogic() for _ in range(num_envs)]
        self.visual_game = SnakeGameVisual()
        metadata = self.agent.model.load(optimizer=self.agent.trainer.optimizer)
        self.agent.epsilon = metadata.get('epsilon', self.agent.epsilon)
        self.agent.n_games = metadata.get('n_games', 0)
        self.record = metadata.get('record', 0)

    def train(self, visual=True, plotting=True):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0

        while True:
            for i in range(self.num_envs):
                game = self.games[i]
                state_old = self.agent.get_state(game)
                final_move = self.agent.get_action(state_old)
                reward, done, score = game.play_step(final_move)
                state_new = self.agent.get_state(game)
                self.agent.train_short_memory(state_old, final_move, reward, state_new, done)
                self.agent.remember(state_old, final_move, reward, state_new, done)

                if done:
                    game.reset()
                    self.agent.n_games += 1
                    self.agent.train_long_memory()

                    if score > self.record:
                        self.record = score
                        metadata = {
                            'epsilon': self.agent.epsilon,
                            'record': self.record,
                            'n_games': self.agent.n_games
                        }
                        self.agent.model.save(optimizer=self.agent.trainer.optimizer, metadata=metadata)
                    print('Game', self.agent.n_games, 'Score', score, 'Record:', self.record)
                    
                    if plotting == True:
                        plot_scores.append(score)
                        total_score += score
                        mean_score = total_score / self.agent.n_games
                        plot_mean_scores.append(mean_score)
                        plot(plot_scores, plot_mean_scores)

            # Run one visual game
            if visual == True:
                self.run_visual()

    def run_visual(self):
        state_old = self.agent.get_state(self.visual_game)
        final_move = self.agent.get_action(state_old)
        reward, done, score = self.visual_game.play_step(final_move)
        state_new = self.agent.get_state(self.visual_game)
        self.agent.train_short_memory(state_old, final_move, reward, state_new, done)
        self.agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            self.visual_game.reset()
            self.agent.n_games += 1
            self.agent.train_long_memory()

            if score > self.record:
                self.record = score
                metadata = {
                    'epsilon': self.agent.epsilon,
                    'record': self.record,
                    'n_games': self.agent.n_games
                }
                self.agent.model.save(optimizer=self.agent.trainer.optimizer, metadata=metadata)