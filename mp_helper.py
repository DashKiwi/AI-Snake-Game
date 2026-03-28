import numpy as np
import torch
import random


def _worker_run_episodes(args):
    from game import SnakeGameLogic, Direction, Point
    from model import Linear_QNet
    import torch

    weights_numpy, epsilon, n_games, num_episodes = args

    model = Linear_QNet(11, 256, 3)
    state_dict = {k: torch.tensor(v) for k, v in weights_numpy.items()}
    model.load_state_dict(state_dict)
    model.eval()

    def get_state(game):
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

    def get_action(state):
        final_move = [0, 0, 0]
        if random.randint(0, 200) < (80 - n_games):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    results = []
    for _ in range(num_episodes):
        game = SnakeGameLogic()
        game.n_games = n_games
        done = False
        experiences = []

        while not done:
            state_old = get_state(game)
            action = get_action(state_old)
            reward, done, score = game.play_step(action)
            state_new = get_state(game)
            experiences.append((state_old, action, reward, state_new, done))

        results.append((experiences, score))

    return results

def run_parallel_episodes(num_envs, n_workers, weights, epsilon, n_games, mp_context):
    episodes_per_worker = [num_envs // n_workers] * n_workers
    for i in range(num_envs % n_workers):
        episodes_per_worker[i] += 1 

    args_list = [
        (weights, epsilon, n_games, n_eps)
        for n_eps in episodes_per_worker
        if n_eps > 0
    ]

    with mp_context.Pool(processes=n_workers) as pool:
        worker_results = pool.map(_worker_run_episodes, args_list)

    all_experiences = []
    all_scores = []
    for batch in worker_results:
        for experiences, score in batch:
            all_experiences.append(experiences)
            all_scores.append(score)

    return all_experiences, all_scores