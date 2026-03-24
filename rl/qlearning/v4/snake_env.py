"""
Snake environment v4 — maximum tabular Q-learning state (22 bits).

State (22 binary floats):
    [0-2]   danger dist 1  (straight, right, left)
    [3-5]   danger dist 2  (straight, right, left)
    [6-7]   danger diag    (fwd-right, fwd-left)
    [8-11]  direction      (up, down, left, right)
    [12-15] apple          (up, down, left, right)
    [16-18] flood-fill safe (straight, right, left)
            → 1.0 if reachable space >= snake length
    [19]    danger dist 3 straight
    [20-21] danger rear diag (back-right, back-left)

Grid: 10 wide × 9 tall.
"""

import random
import math
from collections import deque

# ── Cell types ────────────────────────────────────────────────────────────────
EMPTY      = 0
SNAKE_HEAD = 1
SNAKE_BODY = 2
APPLE      = 3
WALL       = 4

# ── Actions ───────────────────────────────────────────────────────────────────
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

DIRECTION = {
    UP:    ( 0, -1),
    DOWN:  ( 0,  1),
    LEFT:  (-1,  0),
    RIGHT: ( 1,  0),
}


class SnakeEnv:
    """
    Google Snake environment v4 — 22-feature state for tabular Q-learning.

    New features over v3:
      - Flood-fill spatial awareness (3 bits): detects dead ends
      - Danger at distance 3 straight (1 bit): deeper forward vision
      - Rear diagonal danger (2 bits): 360° awareness
    """

    def __init__(self, grid_w=10, grid_h=9, n_apples=5):
        self.grid_w   = grid_w
        self.grid_h   = grid_h
        self.n_apples = n_apples

        self.snake             = []
        self.dx                = 0
        self.dy                = 0
        self.apples            = []
        self.walls             = []
        self.apples_eaten      = 0
        self.steps             = 0
        self.steps_since_apple = 0
        self.done              = False

        self.grace    = 20
        self.pen_base = 0.001
        self.pen_rate = 0.05
        self.max_steps = grid_w * grid_h * 4

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        self.snake = [(2, 4), (1, 4), (0, 4)]
        self.dx, self.dy = 1, 0
        self.apples       = [(4,2), (8,2), (6,4), (4,6), (8,6)]
        self.walls        = []
        self.apples_eaten      = 0
        self.steps             = 0
        self.steps_since_apple = 0
        self.done              = False
        return self._get_state()

    def step(self, action):
        assert not self.done, "Call reset() before stepping after game over."
        self.steps += 1

        prev_dx, prev_dy = self.dx, self.dy

        new_dx, new_dy = DIRECTION[action]
        if not (new_dx == -self.dx and new_dy == -self.dy):
            self.dx, self.dy = new_dx, new_dy

        hx, hy = self.snake[0]
        new_head = (hx + self.dx, hy + self.dy)

        if self._is_fatal(new_head):
            safe_exits = sum(
                1 for ddx, ddy in DIRECTION.values()
                if not (ddx == -prev_dx and ddy == -prev_dy)
                and not self._is_fatal((hx + ddx, hy + ddy))
            )
            penalty = -1.0 if safe_exits == 0 else -5.0
            self.done = True
            return self._get_state(), penalty, True

        self.snake.insert(0, new_head)
        self.steps_since_apple += 1

        if new_head in self.apples:
            self.apples.remove(new_head)
            self.apples_eaten     += 1
            self.steps_since_apple = 0
            reward = 1.0
            if self.apples_eaten == 1 or (self.apples_eaten - 1) % 2 == 0:
                self._spawn_wall()
            self._spawn_apples()
        else:
            self.snake.pop()
            over_grace = max(0, self.steps_since_apple - self.grace)
            reward = -min(self.pen_base * math.exp(self.pen_rate * over_grace), 0.5)

        if self.steps >= self.max_steps:
            self.done = True
            return self._get_state(), reward, True

        return self._get_state(), reward, False

    # ── State representation (22 bits) ────────────────────────────────────────

    def _get_state(self):
        hx, hy = self.snake[0]

        straight = (self.dx,       self.dy)
        right    = (-self.dy,      self.dx)
        left     = ( self.dy,     -self.dx)
        back     = (-self.dx,     -self.dy)

        # ── Danger dist 1 ────────────────────────────────────────────────
        danger_s1 = 1.0 if self._is_fatal((hx + straight[0], hy + straight[1])) else 0.0
        danger_r1 = 1.0 if self._is_fatal((hx + right[0],    hy + right[1]))    else 0.0
        danger_l1 = 1.0 if self._is_fatal((hx + left[0],     hy + left[1]))     else 0.0

        # ── Danger dist 2 ────────────────────────────────────────────────
        danger_s2 = 1.0 if self._is_fatal((hx + 2*straight[0], hy + 2*straight[1])) else 0.0
        danger_r2 = 1.0 if self._is_fatal((hx + 2*right[0],    hy + 2*right[1]))    else 0.0
        danger_l2 = 1.0 if self._is_fatal((hx + 2*left[0],     hy + 2*left[1]))     else 0.0

        # ── Diagonal danger (forward) ────────────────────────────────────
        danger_diag_r = 1.0 if self._is_fatal((hx + straight[0] + right[0],
                                                hy + straight[1] + right[1])) else 0.0
        danger_diag_l = 1.0 if self._is_fatal((hx + straight[0] + left[0],
                                                hy + straight[1] + left[1]))  else 0.0

        # ── Direction one-hot ────────────────────────────────────────────
        dir_up    = 1.0 if (self.dx, self.dy) == ( 0, -1) else 0.0
        dir_down  = 1.0 if (self.dx, self.dy) == ( 0,  1) else 0.0
        dir_left  = 1.0 if (self.dx, self.dy) == (-1,  0) else 0.0
        dir_right = 1.0 if (self.dx, self.dy) == ( 1,  0) else 0.0

        # ── Apple direction (absolute) ───────────────────────────────────
        ax, ay = self._nearest_apple()
        apple_up    = 1.0 if ay < hy else 0.0
        apple_down  = 1.0 if ay > hy else 0.0
        apple_left  = 1.0 if ax < hx else 0.0
        apple_right = 1.0 if ax > hx else 0.0

        # ── Flood-fill safety (NEW) ──────────────────────────────────────
        # 1.0 if going that direction leads to enough space for the snake
        snake_len = len(self.snake)
        pos_s = (hx + straight[0], hy + straight[1])
        pos_r = (hx + right[0],    hy + right[1])
        pos_l = (hx + left[0],     hy + left[1])

        space_s = 1.0 if self._flood_count(pos_s) >= snake_len else 0.0
        space_r = 1.0 if self._flood_count(pos_r) >= snake_len else 0.0
        space_l = 1.0 if self._flood_count(pos_l) >= snake_len else 0.0

        # ── Danger dist 3 straight (NEW) ─────────────────────────────────
        danger_s3 = 1.0 if self._is_fatal((hx + 3*straight[0], hy + 3*straight[1])) else 0.0

        # ── Rear diagonal danger (NEW) ───────────────────────────────────
        danger_back_r = 1.0 if self._is_fatal((hx + back[0] + right[0],
                                                hy + back[1] + right[1])) else 0.0
        danger_back_l = 1.0 if self._is_fatal((hx + back[0] + left[0],
                                                hy + back[1] + left[1]))  else 0.0

        return [
            danger_s1, danger_r1, danger_l1,           # 0-2
            danger_s2, danger_r2, danger_l2,           # 3-5
            danger_diag_r, danger_diag_l,              # 6-7
            dir_up, dir_down, dir_left, dir_right,     # 8-11
            apple_up, apple_down, apple_left, apple_right,  # 12-15
            space_s, space_r, space_l,                 # 16-18
            danger_s3,                                 # 19
            danger_back_r, danger_back_l,              # 20-21
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_fatal(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return True
        if pos in self.walls:
            return True
        if pos in self.snake:
            return True
        return False

    def _flood_count(self, start):
        """
        BFS flood fill from start position.
        Returns number of reachable cells.
        Early-terminates once count >= snake length (enough space = safe).
        """
        if self._is_fatal(start):
            return 0
        snake_set = set(self.snake)
        wall_set  = set(self.walls)
        target    = len(self.snake)
        visited   = {start}
        queue     = deque([start])
        while queue:
            if len(visited) >= target:
                return len(visited)    # early exit: enough space
            x, y = queue.popleft()
            for ddx, ddy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = x + ddx, y + ddy
                if (nx, ny) not in visited and 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    if (nx, ny) not in snake_set and (nx, ny) not in wall_set:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return len(visited)

    def _nearest_apple(self):
        if not self.apples:
            return (self.grid_w // 2, self.grid_h // 2)
        hx, hy = self.snake[0]
        return min(self.apples, key=lambda a: abs(a[0] - hx) + abs(a[1] - hy))

    def _spawn_apples(self):
        occupied = set(self.snake) | set(self.walls) | set(self.apples)
        attempts = 0
        while len(self.apples) < self.n_apples and attempts < 200:
            attempts += 1
            candidate = (random.randint(0, self.grid_w - 1), random.randint(0, self.grid_h - 1))
            if candidate not in occupied:
                self.apples.append(candidate)
                occupied.add(candidate)

    def _spawn_wall(self):
        if len(self.walls) >= 17:
            return
        snake_set = set(self.snake)
        apple_set = set(self.apples)
        wall_set  = set(self.walls)
        hx, hy    = self.snake[0]
        for _ in range(100):
            x = random.randint(0, self.grid_w - 1)
            y = random.randint(0, self.grid_h - 1)
            if (x, y) in snake_set or (x, y) in apple_set:
                continue
            if any(abs(wx - x) <= 1 and abs(wy - y) <= 1 for wx, wy in wall_set):
                continue
            if abs(x - hx) + abs(y - hy) <= 3:
                continue
            corners = [(0,0),(self.grid_w-1,0),(0,self.grid_h-1),(self.grid_w-1,self.grid_h-1)]
            if any(abs(x-cx)<=1 and abs(y-cy)<=1 for cx,cy in corners):
                continue
            self.walls.append((x, y))
            break

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_grid(self):
        grid = [[EMPTY] * self.grid_w for _ in range(self.grid_h)]
        for wx, wy in self.walls:
            grid[wy][wx] = WALL
        for ax, ay in self.apples:
            grid[ay][ax] = APPLE
        for i, (sx, sy) in enumerate(self.snake):
            grid[sy][sx] = SNAKE_HEAD if i == 0 else SNAKE_BODY
        return grid

    def print_grid(self):
        symbols = {EMPTY: ".", SNAKE_HEAD: "H", SNAKE_BODY: "s", APPLE: "A", WALL: "#"}
        grid = self.get_grid()
        print("+" + "-" * self.grid_w + "+")
        for row in grid:
            print("|" + "".join(symbols[c] for c in row) + "|")
        print("+" + "-" * self.grid_w + "+")
        print(f"Score: {self.apples_eaten}  Step: {self.steps}  Walls: {len(self.walls)}")

    def render(self, cell_size=60, fps=10):
        import pygame
        if not hasattr(self, '_screen'):
            pygame.init()
            w = self.grid_w * cell_size
            h = self.grid_h * cell_size + 40
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Snake RL v4")
            self._clock  = pygame.time.Clock()
            self._font   = pygame.font.SysFont("Arial", 20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                color = (170, 215, 81) if (x + y) % 2 == 0 else (162, 209, 73)
                pygame.draw.rect(self._screen, color,
                                 (x * cell_size, y * cell_size, cell_size, cell_size))
        for wx, wy in self.walls:
            pygame.draw.rect(self._screen, (83, 138, 52),
                             (wx * cell_size, wy * cell_size, cell_size, cell_size))
            pygame.draw.rect(self._screen, (60, 110, 35),
                             (wx * cell_size + 6, wy * cell_size + 6,
                              cell_size - 12, cell_size - 12))
        for ax, ay in self.apples:
            cx = ax * cell_size + cell_size // 2
            cy = ay * cell_size + cell_size // 2
            pygame.draw.circle(self._screen, (220, 50, 50), (cx, cy), cell_size // 2 - 4)
        for i, (sx, sy) in enumerate(self.snake):
            color = (30, 90, 180) if i == 0 else (75, 133, 212)
            pygame.draw.rect(self._screen, color,
                             (sx * cell_size + 2, sy * cell_size + 2,
                              cell_size - 4, cell_size - 4),
                             border_radius=8)
        bar_y = self.grid_h * cell_size
        pygame.draw.rect(self._screen, (40, 40, 40),
                         (0, bar_y, self.grid_w * cell_size, 40))
        text = self._font.render(
            f"Score: {self.apples_eaten}   Steps: {self.steps}   Walls: {len(self.walls)}",
            True, (255, 255, 255))
        self._screen.blit(text, (10, bar_y + 10))
        pygame.display.flip()
        self._clock.tick(fps)


if __name__ == "__main__":
    env = SnakeEnv()
    state = env.reset()
    print(f"State length : {len(state)}  (expected 22)")
    print(f"State values : {state}")
    env.print_grid()

    for i in range(10):
        action = random.randint(0, 3)
        state, reward, done = env.step(action)
        print(f"Step {i+1}: action={action}  reward={reward:.3f}  done={done}")
        if done:
            print("Episode ended.")
            break
    print("\nEnv is working correctly.")
