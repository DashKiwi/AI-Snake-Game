import random
from enum import Enum
from collections import namedtuple
import numpy as np

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 30

SHAPING_DECAY_GAMES = 200


class SnakeGameLogic:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.n_games = 0
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.last_distance_to_food = self._dist_to_food()

    def _dist_to_food(self):
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _shaping_scale(self):
        return max(0.0, 1.0 - self.n_games / SHAPING_DECAY_GAMES)

    def play_step(self, action):
        self.frame_iteration += 1
        game_over = False

        self._move(action)
        self.snake.insert(0, self.head)

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            self.snake.pop()
            length_penalty = -10
            return length_penalty, game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
            self.frame_iteration = 0
            self.last_distance_to_food = self._dist_to_food()
            return 10, game_over, self.score

        self.snake.pop()
        reward = -0.1
        scale = self._shaping_scale()
        if scale > 0:
            dist_now = self._dist_to_food()
            shaping = 0.5 if dist_now < self.last_distance_to_food else -0.5
            reward += shaping * scale
            self.last_distance_to_food = dist_now

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


class SnakeGameVisual(SnakeGameLogic):
    def __init__(self, w=640, h=480, speed=SPEED):
        import pygame
        pygame.init()
        self.pygame = pygame

        self.COLOR1 = (0, 255, 0)
        self.COLOR2 = (153, 255, 18)
        self.RED    = (200, 0, 0)
        self.WHITE  = (255, 255, 255)
        self.BLACK  = (0, 0, 0)

        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.speed = speed
        super().__init__(w, h)

    def play_step(self, action):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        reward, game_over, score = super().play_step(action)
        self._update_ui()
        self.clock.tick(self.speed)
        return reward, game_over, score

    def _update_ui(self):
        pygame = self.pygame
        self.display.fill(self.BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, self.COLOR1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, self.COLOR2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, self.RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render("Score: " + str(self.score), True, self.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()