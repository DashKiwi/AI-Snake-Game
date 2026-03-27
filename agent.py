import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameLogic, SnakeGameVisual, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from mp_helper import run_parallel_episodes

import multiprocessing as mp
import threading

MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
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
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
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

    def get_weights(self):
        """Return model weights as plain numpy arrays (picklable)."""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}


class VectorizedAgent:
    def __init__(self, num_envs=64):
        self.num_envs = num_envs
        self.agent = Agent()
        self.record = 0
        self.mp_context = mp.get_context('spawn')

        metadata = self.agent.model.load(optimizer=self.agent.trainer.optimizer)
        self.agent.epsilon = metadata.get('epsilon', self.agent.epsilon)
        self.agent.n_games = metadata.get('n_games', 0)
        self.record = metadata.get('record', 0)

    def train(self, visual=True, plotting=True):
        plot_scores = []
        plot_mean_scores = []

        n_workers = max(1, mp.cpu_count() - 1)
        print(f"Using {n_workers} parallel worker processes for {self.num_envs} environments.")

        if visual:
            visual_game = SnakeGameVisual()
            stop_event = threading.Event()

            train_thread = threading.Thread(
                target=self._training_loop,
                args=(n_workers, plot_scores, plot_mean_scores, plotting, stop_event),
                daemon=True
            )
            train_thread.start()

            while not stop_event.is_set():
                state = self.agent.get_state(visual_game)
                action = self.agent.get_action(state)
                _, done, _ = visual_game.play_step(action)
                if done:
                    visual_game.reset()
        else:
            stop_event = threading.Event()
            self._training_loop(n_workers, plot_scores, plot_mean_scores, plotting, stop_event)

    def _training_loop(self, n_workers, plot_scores, plot_mean_scores, plotting, stop_event):
        total_score = 0
        try:
            while True:
                weights = self.agent.get_weights()
                epsilon = self.agent.epsilon
                n_games = self.agent.n_games

                all_experiences, all_scores = run_parallel_episodes(
                    num_envs=self.num_envs,
                    n_workers=n_workers,
                    weights=weights,
                    epsilon=epsilon,
                    n_games=n_games,
                    mp_context=self.mp_context
                )

                for experiences, score in zip(all_experiences, all_scores):
                    for (state_old, final_move, reward, state_new, done) in experiences:
                        self.agent.remember(state_old, final_move, reward, state_new, done)
                        self.agent.train_short_memory(state_old, final_move, reward, state_new, done)

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

                    print(f'Game {self.agent.n_games}  Score {score}  Record: {self.record}')

                    if plotting:
                        plot_scores.append(score)
                        total_score += score
                        mean_score = total_score / self.agent.n_games
                        plot_mean_scores.append(mean_score)
                        plot(plot_scores, plot_mean_scores)

        except Exception as e:
            print(f"Training thread error: {e}")
        finally:
            stop_event.set()