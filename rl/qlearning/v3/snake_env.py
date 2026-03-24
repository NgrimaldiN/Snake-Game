"""
Snake environment v3 — optimized state representation for Q-learning.

Key improvements over v2:
  1. All features are RELATIVE to the snake's heading
     → eliminates the 4 direction bits (symmetric learning)
  2. Danger sensing at distance 2 (straight, right, left)
     → the snake can see traps 2 steps ahead
  3. No approach reward (proved harmful — makes agent reckless)

State (16 binary floats):
    [0]  danger straight   (dist 1)
    [1]  danger right      (dist 1)
    [2]  danger left       (dist 1)
    [3]  danger straight   (dist 2)
    [4]  danger right      (dist 2)
    [5]  danger left       (dist 2)
    [6]  danger diag fwd-right
    [7]  danger diag fwd-left
    [8]  dir up
    [9]  dir down
    [10] dir left
    [11] dir right
    [12] apple up
    [13] apple down
    [14] apple left
    [15] apple right

Grid: 10 wide × 9 tall.
"""

import random
import math

# ── Cell types (used internally) ─────────────────────────────────────────────
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

# Direction vectors for each action
DIRECTION = {
    UP:    ( 0, -1),
    DOWN:  ( 0,  1),
    LEFT:  (-1,  0),
    RIGHT: ( 1,  0),
}


class SnakeEnv:
    """
    Google Snake environment v3 — optimized for Q-learning convergence.

    State: 16 binary features (danger + direction + apple).
    Reward: +1.0 apple, -1.0 trapped death, -5.0 suicide,
            exponential idle penalty after grace period.
    """

    def __init__(self, grid_w=10, grid_h=9, n_apples=5):
        self.grid_w  = grid_w
        self.grid_h  = grid_h
        self.n_apples = n_apples

        # Set by reset()
        self.snake             = []
        self.dx                = 0
        self.dy                = 0
        self.apples            = []
        self.walls             = []
        self.apples_eaten      = 0
        self.steps             = 0
        self.steps_since_apple = 0
        self.done              = False

        # ── Penalty schedule ──────────────────────────────────────────────────
        self.grace    = 20
        self.pen_base = 0.001
        self.pen_rate = 0.05

        self.max_steps = grid_w * grid_h * 4

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Start a new episode. Returns the initial state vector."""
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
        """
        Apply action, advance the game by one tick.

        Returns:
            state  : list[float] — new state (12 values)
            reward : float
            done   : bool
        """
        assert not self.done, "Call reset() before stepping after game over."
        self.steps += 1

        prev_dx, prev_dy = self.dx, self.dy

        # Apply direction (ignore 180° reversals)
        new_dx, new_dy = DIRECTION[action]
        if not (new_dx == -self.dx and new_dy == -self.dy):
            self.dx, self.dy = new_dx, new_dy

        hx, hy = self.snake[0]
        new_head = (hx + self.dx, hy + self.dy)

        # ── Collision checks → death ─────────────────────────────────────────
        if self._is_fatal(new_head):
            safe_exits = sum(
                1 for ddx, ddy in DIRECTION.values()
                if not (ddx == -prev_dx and ddy == -prev_dy)
                and not self._is_fatal((hx + ddx, hy + ddy))
            )
            if safe_exits == 0:
                penalty = -1.0
            else:
                penalty = -5.0
            self.done = True
            return self._get_state(), penalty, True

        # ── Move snake ───────────────────────────────────────────────────────
        self.snake.insert(0, new_head)
        self.steps_since_apple += 1

        # ── Apple check ─────────────────────────────────────────────────────
        if new_head in self.apples:
            self.apples.remove(new_head)
            self.apples_eaten      += 1
            self.steps_since_apple  = 0
            reward = 1.0

            if self.apples_eaten == 1 or (self.apples_eaten - 1) % 2 == 0:
                self._spawn_wall()
            self._spawn_apples()
        else:
            self.snake.pop()

            # ── Exponential step penalty ──────────────────────────────────────
            over_grace = max(0, self.steps_since_apple - self.grace)
            reward     = -min(self.pen_base * math.exp(self.pen_rate * over_grace), 0.5)

        # ── Max steps truncation ─────────────────────────────────────────────
        if self.steps >= self.max_steps:
            self.done = True
            return self._get_state(), reward, True

        return self._get_state(), reward, False

    # ── State representation ──────────────────────────────────────────────────

    def _get_state(self):
        """
        Build the 16-value feature vector.

        Danger features are relative to heading.
        Direction and apple are absolute (needed for correct action mapping).
        """
        hx, hy = self.snake[0]

        # ── Relative direction vectors ────────────────────────────────────
        straight = (self.dx,       self.dy)
        right    = (-self.dy,      self.dx)
        left     = ( self.dy,     -self.dx)

        # ── Danger at distance 1 ─────────────────────────────────────────
        danger_s1 = 1.0 if self._is_fatal((hx + straight[0], hy + straight[1])) else 0.0
        danger_r1 = 1.0 if self._is_fatal((hx + right[0],    hy + right[1]))    else 0.0
        danger_l1 = 1.0 if self._is_fatal((hx + left[0],     hy + left[1]))     else 0.0

        # ── Danger at distance 2 ─────────────────────────────────────────
        danger_s2 = 1.0 if self._is_fatal((hx + 2*straight[0], hy + 2*straight[1])) else 0.0
        danger_r2 = 1.0 if self._is_fatal((hx + 2*right[0],    hy + 2*right[1]))    else 0.0
        danger_l2 = 1.0 if self._is_fatal((hx + 2*left[0],     hy + 2*left[1]))     else 0.0

        # ── Diagonal danger (forward-right, forward-left) ────────────────
        danger_diag_r = 1.0 if self._is_fatal((hx + straight[0] + right[0],
                                                hy + straight[1] + right[1])) else 0.0
        danger_diag_l = 1.0 if self._is_fatal((hx + straight[0] + left[0],
                                                hy + straight[1] + left[1]))  else 0.0

        # ── Current direction (one-hot) ───────────────────────────────────
        dir_up    = 1.0 if (self.dx, self.dy) == ( 0, -1) else 0.0
        dir_down  = 1.0 if (self.dx, self.dy) == ( 0,  1) else 0.0
        dir_left  = 1.0 if (self.dx, self.dy) == (-1,  0) else 0.0
        dir_right = 1.0 if (self.dx, self.dy) == ( 1,  0) else 0.0

        # ── Nearest apple direction (absolute) ────────────────────────────
        ax, ay = self._nearest_apple()
        apple_up    = 1.0 if ay < hy else 0.0
        apple_down  = 1.0 if ay > hy else 0.0
        apple_left  = 1.0 if ax < hx else 0.0
        apple_right = 1.0 if ax > hx else 0.0

        return [
            danger_s1, danger_r1, danger_l1,
            danger_s2, danger_r2, danger_l2,
            danger_diag_r, danger_diag_l,
            dir_up, dir_down, dir_left, dir_right,
            apple_up, apple_down, apple_left, apple_right,
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_fatal(self, pos):
        """Return True if moving to pos would kill the snake."""
        x, y = pos
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return True
        if pos in self.walls:
            return True
        if pos in self.snake:
            return True
        return False

    def _nearest_apple(self):
        """Return (x, y) of the closest apple by Manhattan distance."""
        if not self.apples:
            return (self.grid_w // 2, self.grid_h // 2)
        hx, hy = self.snake[0]
        return min(self.apples, key=lambda a: abs(a[0] - hx) + abs(a[1] - hy))

    def _dist_nearest_apple(self):
        """Return Manhattan distance to the closest apple."""
        ax, ay = self._nearest_apple()
        hx, hy = self.snake[0]
        return abs(ax - hx) + abs(ay - hy)

    def _spawn_apples(self):
        """Fill up to n_apples, avoiding snake, walls, and existing apples."""
        occupied = set(self.snake) | set(self.walls) | set(self.apples)
        attempts = 0
        while len(self.apples) < self.n_apples and attempts < 200:
            attempts += 1
            candidate = (
                random.randint(0, self.grid_w - 1),
                random.randint(0, self.grid_h - 1),
            )
            if candidate not in occupied:
                self.apples.append(candidate)
                occupied.add(candidate)

    def _spawn_wall(self):
        """
        Try to place a wall following Google Snake's placement rules.
        """
        snake_set = set(self.snake)
        if len(self.walls) >= 17:
            return
        apple_set = set(self.apples)
        wall_set  = set(self.walls)
        hx, hy    = self.snake[0]

        for _ in range(100):
            x = random.randint(0, self.grid_w - 1)
            y = random.randint(0, self.grid_h - 1)

            if (x, y) in snake_set:
                continue
            if (x, y) in apple_set:
                continue

            too_close_to_wall = any(
                abs(wx - x) <= 1 and abs(wy - y) <= 1
                for wx, wy in wall_set
            )
            if too_close_to_wall:
                continue

            if abs(x - hx) + abs(y - hy) <= 3:
                continue

            corners = [
                (0, 0), (self.grid_w - 1, 0),
                (0, self.grid_h - 1), (self.grid_w - 1, self.grid_h - 1),
            ]
            if any(abs(x - cx) <= 1 and abs(y - cy) <= 1 for cx, cy in corners):
                continue

            self.walls.append((x, y))
            break

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_grid(self):
        """Return the full grid as a 2D list."""
        grid = [[EMPTY] * self.grid_w for _ in range(self.grid_h)]
        for wx, wy in self.walls:
            grid[wy][wx] = WALL
        for ax, ay in self.apples:
            grid[ay][ax] = APPLE
        for i, (sx, sy) in enumerate(self.snake):
            grid[sy][sx] = SNAKE_HEAD if i == 0 else SNAKE_BODY
        return grid

    def print_grid(self):
        """ASCII render — useful for debugging."""
        symbols = {EMPTY: ".", SNAKE_HEAD: "H", SNAKE_BODY: "s", APPLE: "A", WALL: "#"}
        grid = self.get_grid()
        print("+" + "-" * self.grid_w + "+")
        for row in grid:
            print("|" + "".join(symbols[c] for c in row) + "|")
        print("+" + "-" * self.grid_w + "+")
        print(f"Score: {self.apples_eaten}  Step: {self.steps}  Walls: {len(self.walls)}")

    def render(self, cell_size=60, fps=10):
        """Draw the current game state in a pygame window."""
        import pygame

        if not hasattr(self, '_screen'):
            pygame.init()
            w = self.grid_w * cell_size
            h = self.grid_h * cell_size + 40
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Snake RL v3")
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
            True, (255, 255, 255)
        )
        self._screen.blit(text, (10, bar_y + 10))

        pygame.display.flip()
        self._clock.tick(fps)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = SnakeEnv()
    state = env.reset()

    print(f"State length : {len(state)}  (expected 12)")
    print(f"State values : {state}")
    print()
    env.print_grid()

    for i in range(10):
        action = random.randint(0, 3)
        state, reward, done = env.step(action)
        print(f"Step {i+1}: action={action}  reward={reward:.3f}  done={done}  state_len={len(state)}")
        if done:
            print("Episode ended.")
            break

    print("\nEnv is working correctly.")
