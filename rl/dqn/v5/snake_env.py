"""
Snake environment v5 — raw grid state for CNN-based DQN.

State: numpy array, shape (8, H, W), dtype uint8.
    Channel 0 : snake head   (1 at head cell)
    Channel 1 : snake body   (1 at each body cell)
    Channel 2 : apple        (1 at each apple cell)
    Channel 3 : wall         (1 at each wall cell)
    Channel 4-7 : direction  (one-hot — the active direction channel is all-1s)

Why 8 binary channels instead of one integer per cell?
    Cell types are categorical (apple ≠ 2 × head).  One-hot channels let the
    CNN treat each entity type independently — same idea as one-hot encoding
    in any ML pipeline.  Direction is broadcast across the full grid so every
    conv filter can "see" it regardless of receptive field.

Grid: 10 wide × 9 tall.
"""

import random
import math
import numpy as np
from collections import deque

# ── Cell types (for rendering only) ──────────────────────────────────────────
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

DIR_CHANNEL = {
    ( 0, -1): 4,   # up
    ( 0,  1): 5,   # down
    (-1,  0): 6,   # left
    ( 1,  0): 7,   # right
}


class SnakeEnv:
    """
    Google Snake environment v5 — 8-channel grid state for DQN.

    Same game rules and reward shaping as v4 (loop detection, exponential
    idle penalty, avoidable-death vs trapped-death distinction).
    Only the state representation changes: raw grid → CNN input.
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
        self.max_steps = 2000

        self.penalty_trap_death     = -1.0
        self.penalty_avoidable_death = -5.0
        self.penalty_loop           = self.penalty_avoidable_death
        self.loop_max_visits_per_cell = 2

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        self.snake = [(2, 4), (1, 4), (0, 4)]
        self.dx, self.dy = 1, 0
        self.apples       = [(4, 2), (8, 2), (6, 4), (4, 6), (8, 6)]
        self.walls        = []
        self.apples_eaten      = 0
        self.steps             = 0
        self.steps_since_apple = 0
        self.done              = False
        self._head_visit_count_since_apple = {self.snake[0]: 1}
        self._init_wall_block_grid()
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

        # ── Collision death ───────────────────────────────────────────────
        if self._is_fatal(new_head):
            safe_exits = sum(
                1 for ddx, ddy in DIRECTION.values()
                if not (ddx == -prev_dx and ddy == -prev_dy)
                and not self._is_fatal((hx + ddx, hy + ddy))
            )
            penalty = self.penalty_trap_death if safe_exits == 0 else self.penalty_avoidable_death
            self.done = True
            return self._get_state(), penalty, True

        # ── Loop death (3rd visit to same cell since last apple) ──────────
        prev_visits = self._head_visit_count_since_apple.get(new_head, 0)
        if prev_visits >= self.loop_max_visits_per_cell:
            self.done = True
            return self._get_state(), self.penalty_loop, True

        # ── Move ──────────────────────────────────────────────────────────
        self.snake.insert(0, new_head)
        self.steps_since_apple += 1

        if new_head in self.apples:
            self.apples.remove(new_head)
            self.apples_eaten     += 1
            self.steps_since_apple = 0
            reward = 1.0
            self._head_visit_count_since_apple = {new_head: 1}
            if self.apples_eaten == 1 or (self.apples_eaten - 1) % 2 == 0:
                self._spawn_wall()
            self._spawn_apples()
        else:
            self.snake.pop()
            self._head_visit_count_since_apple[new_head] = prev_visits + 1
            over_grace = max(0, self.steps_since_apple - self.grace)
            reward = -min(self.pen_base * math.exp(self.pen_rate * over_grace), 0.5)

        if self.steps >= self.max_steps:
            self.done = True
            return self._get_state(), reward, True

        return self._get_state(), reward, False

    # ── State representation (8-channel grid) ─────────────────────────────────

    def _get_state(self):
        state = np.zeros((8, self.grid_h, self.grid_w), dtype=np.uint8)

        hx, hy = self.snake[0]
        state[0, hy, hx] = 1

        for sx, sy in self.snake[1:]:
            state[1, sy, sx] = 1

        for ax, ay in self.apples:
            state[2, ay, ax] = 1

        for wx, wy in self.walls:
            state[3, wy, wx] = 1

        state[DIR_CHANNEL[(self.dx, self.dy)]] = 1

        return state

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

    def _spawn_apples(self):
        occupied = set(self.snake) | set(self.walls) | set(self.apples)
        while len(self.apples) < self.n_apples:
            candidates = [
                (x, y)
                for x in range(self.grid_w)
                for y in range(self.grid_h)
                if (x, y) not in occupied
            ]
            if not candidates:
                break
            pos = random.choice(candidates)
            self.apples.append(pos)
            occupied.add(pos)

    def _init_wall_block_grid(self):
        """Initialize the blocking grid matching Google Snake's JVD.reset().

        Pre-blocks 8 corner-adjacent cells on the edges — corners themselves
        remain valid for wall placement.
        """
        W, H = self.grid_w, self.grid_h
        self._wblock = [[0] * W for _ in range(H)]
        preblocked = [
            (0, 1), (0, H - 2),
            (W - 1, 1), (W - 1, H - 2),
            (1, 0), (W - 2, 0),
            (1, H - 1), (W - 2, H - 1),
        ]
        for bx, by in preblocked:
            self._wblock[by][bx] = 1

    def _place_wall_block(self, x, y):
        """Mark a wall in the blocking grid: block 8 neighbors + edge extras."""
        W, H = self.grid_w, self.grid_h
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    self._wblock[ny][nx] += 1

        if x == 0 or x == W - 1:
            if y - 2 >= 0:
                self._wblock[y - 2][x] += 1
            if y + 2 <= H - 1:
                self._wblock[y + 2][x] += 1
        if y == 0 or y == H - 1:
            if x - 2 >= 0:
                self._wblock[y][x - 2] += 1
            if x + 2 <= W - 1:
                self._wblock[y][x + 2] += 1

        cx_pairs = [
            (0, 2, 2, 0), (W - 3, 0, W - 1, 2),
            (0, H - 3, 2, H - 1), (W - 3, H - 1, W - 1, H - 3),
        ]
        for ax, ay, bx_, by_ in cx_pairs:
            if (x == ax and y == ay) or (x == bx_ and y == by_):
                self._wblock[ay][ax] += 1
                self._wblock[by_][bx_] += 1

    def _spawn_wall(self):
        if len(self.walls) >= 17:
            return
        hx, hy = self.snake[0]
        snake_set = set(self.snake)
        apple_set = set(self.apples)

        candidates = []
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if self._wblock[y][x] > 0:
                    continue
                if (x, y) in snake_set or (x, y) in apple_set:
                    continue
                if abs(x - hx) + abs(y - hy) <= 3:
                    continue
                candidates.append((x, y))

        if candidates:
            pos = random.choice(candidates)
            self.walls.append(pos)
            self._place_wall_block(pos[0], pos[1])

    # ── Rendering ─────────────────────────────────────────────────────────────

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

    def render(self, cell_size=60, fps=10, title=None):
        import pygame
        if not hasattr(self, '_screen'):
            pygame.init()
            w = self.grid_w * cell_size
            h = self.grid_h * cell_size + 40
            self._screen = pygame.display.set_mode((w, h))
            self._clock  = pygame.time.Clock()
            self._font   = pygame.font.SysFont("Arial", 20)
        pygame.display.set_caption(title or "Snake DQN v5")
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
    print(f"State shape: {state.shape}  dtype: {state.dtype}  (expected (8, 9, 10) uint8)")
    print(f"Channels with any 1s: {[i for i in range(8) if state[i].any()]}")
    env.print_grid()

    for i in range(10):
        action = random.randint(0, 3)
        state, reward, done = env.step(action)
        print(f"Step {i+1}: action={action}  reward={reward:.3f}  done={done}  shape={state.shape}")
        if done:
            print("Episode ended.")
            break
    print("\nEnv is working correctly.")
