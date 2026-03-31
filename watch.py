import torch
import pygame
from game import SnakeGameVisual, Direction, Point
from model import Linear_QNet
from collections import namedtuple
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_state(game):
    from collections import deque
    
    def flood_fill_size(start_point):
        visited = set()
        queue = deque([start_point])
        body_set = set(game.snake[1:])
        while queue:
            pt = queue.popleft()
            if pt in visited:
                continue
            if pt.x < 0 or pt.x >= game.w or pt.y < 0 or pt.y >= game.h:
                continue
            if pt in body_set:
                continue
            visited.add(pt)
            queue.append(Point(pt.x + 20, pt.y))
            queue.append(Point(pt.x - 20, pt.y))
            queue.append(Point(pt.x, pt.y + 20))
            queue.append(Point(pt.x, pt.y - 20))
        total_cells = (game.w // 20) * (game.h // 20)
        return len(visited) / total_cells

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
        game.food.y > game.head.y,

        flood_fill_size(point_l),
        flood_fill_size(point_r),
        flood_fill_size(point_u),
        flood_fill_size(point_d),
    ]
    return np.array(state, dtype=float)


def get_action(model, state):
    state_tensor = torch.tensor(state, dtype=torch.float).to(device)
    with torch.no_grad():
        prediction = model(state_tensor)
    move = torch.argmax(prediction).item()
    final_move = [0, 0, 0]
    final_move[move] = 1
    return final_move


def watch(speed=15):
    model = Linear_QNet(15, 256, 3).to(device)
    metadata = model.load()

    n_games    = metadata.get('n_games', '?')
    record     = metadata.get('record', '?')

    print(f"Loaded model — trained for {n_games} games, record score: {record}")
    print("Press Q or close the window to quit.")
    print("Press UP/DOWN arrow keys to change speed while watching.")

    model.eval()

    game = SnakeGameVisual(speed=speed)
    games_played = 0
    scores = []

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.speed = min(game.speed + 5, 120)
                if event.key == pygame.K_DOWN:
                    game.speed = max(game.speed - 5, 1)
                if event.key == pygame.K_q:
                    running = False
                    break

        state = get_state(game)
        action = get_action(model, state)
        _, done, score, reason = game.play_step(action)

        if done:
            games_played += 1
            scores.append(score)
            avg = sum(scores) / len(scores)
            print(f"Game {games_played:>4}  |  Score: {score:>4}  |  Avg: {avg:>6.2f}  |  Best: {max(scores):>4}  |  Speed: {game.speed} FPS  |  Reason: {reason}")
            game.reset()

    pygame.quit()


if __name__ == '__main__':
    import os
    os.system('cls||clear')

    print("=== Snake AI Viewer ===")
    try:
        speed = int(input("Playback speed in FPS (default 15, max 120): ") or 15)
    except ValueError:
        speed = 15

    watch(speed=speed)