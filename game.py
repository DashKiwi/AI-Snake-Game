import pygame
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

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
COLOR1 = (0, 255, 0)
COLOR2 = (153, 255, 18)
BLACK = (0,0,0)

BLOCK_SIZE = 20
# Original speed was 40
SPEED = 0

class SnakeGameLogic:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
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
        self.visited = set()
        self.last_distance_to_food = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        reward = -1
        game_over = False

        self._move(action)
        self.snake.insert(0, self.head)

        if self.head not in self.visited:
            self.visited.add(self.head)
            reward += 0.5

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -20
            return reward, game_over, self.score

        current_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        if current_distance < self.last_distance_to_food:
            reward += 0.2
        else:
            reward -= 0.2
        self.last_distance_to_food = current_distance

        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
            self.frame_iteration = 0
            self.visited = set()
        else:
            self.snake.pop()

        if self.frame_iteration > 20 and len(self.visited) < 0.3 * self.frame_iteration:
            reward -= 5

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
    def __init__(self, w=640, h=480):
        pygame.init()
        font = pygame.font.Font('arial.ttf', 25)
        #font = pygame.font.SysFont('arial', 25)s
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font('arial.ttf', 25)
        super().__init__(w, h)

    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        reward, game_over, score = super().play_step(action)
        self._update_ui()
        self.clock.tick(SPEED)  # Adjust visual FPS
        return reward, game_over, score

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, COLOR1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, COLOR2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
