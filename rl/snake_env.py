"""
Snake environment replicating Google Snake (small grid, walls, 5 apples).

Usage:
    env = SnakeEnv()
    state = env.reset()
    state, reward, done = env.step(action)  # action in {0, 1, 2, 3}

Grid: 17 wide x 15 tall (verify against the real game by counting tiles).
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
    Google Snake environment — pure Python, no libraries.

    State (11 floats, values 0.0 or 1.0):
        [0] danger straight      — would die moving forward?
        [1] danger right         — would die turning right?
        [2] danger left          — would die turning left?
        [3] moving up
        [4] moving down
        [5] moving left
        [6] moving right
        [7] apple is up          (relative to head)
        [8] apple is down
        [9] apple is left
        [10] apple is right      (nearest apple)

    Why this representation?
        Simple enough to work with a small MLP.
        Once that works, you can upgrade to the raw grid (17×15×4).

    Reward:
        +1.0   eating an apple
        -1.0   dying
        -0.001 every step        (discourages looping)
    """

    def __init__(self, grid_w=10, grid_h=9, n_apples=5):
        # Google Snake small grid: 10 wide × 9 tall (matches TILE_COUNT_X/Y in snake.js)
        self.grid_w  = grid_w
        self.grid_h  = grid_h
        self.n_apples = n_apples

        # Set by reset()
        self.snake        = []   # list of (x, y), index 0 = head
        self.dx           = 0
        self.dy           = 0
        self.apples       = []   # list of (x, y)
        self.walls        = []   # list of (x, y)
        self.apples_eaten = 0
        self.steps        = 0
        self.done         = False

        # How many steps before we force-stop an episode (prevents infinite loops)
        self.max_steps = grid_w * grid_h * 4

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Start a new episode. Returns the initial state vector."""
        self.snake = [(2, 4), (1, 4), (0, 4)]  # head at index 0, matches JS game starting position
        self.dx, self.dy = 1, 0   # start moving right
        self.apples       = [(4,2), (8,2), (6,4), (4,6), (8,6)]
        self.walls        = []
        self.apples_eaten = 0
        self.steps        = 0
        self.done         = False

        return self._get_state()

    def step(self, action):
        """
        Apply action, advance the game by one tick.

        Args:
            action: int in {UP=0, DOWN=1, LEFT=2, RIGHT=3}

        Returns:
            state  : list[float] — new state (11 values)
            reward : float
            done   : bool — True if the episode ended (death or max_steps)
        """
        assert not self.done, "Call reset() before stepping after game over."
        self.steps += 1

        # Apply direction (ignore 180° reversals — same as Google Snake)
        new_dx, new_dy = DIRECTION[action]
        if not (new_dx == -self.dx and new_dy == -self.dy):
            self.dx, self.dy = new_dx, new_dy

        # Compute new head position
        hx, hy = self.snake[0]
        new_head = (hx + self.dx, hy + self.dy)

        # ── Collision checks → death ─────────────────────────────────────────
        if self._is_fatal(new_head):
            self.done = True
            return self._get_state(), -1.0, True

        # ── Move snake ───────────────────────────────────────────────────────
        self.snake.insert(0, new_head)

        reward = -0.001
        # ── Apple check ─────────────────────────────────────────────────────
        if new_head in self.apples:
            self.apples.remove(new_head)
            self.apples_eaten += 1
            reward = 1.0

            # Google Snake wall spawn pattern:
            # wall after 1st apple, then after every 2nd (3rd, 5th, 7th total)
            if self.apples_eaten == 1 or (self.apples_eaten - 1) % 2 == 0:
                self._spawn_wall()

            self._spawn_apples()   # refill to n_apples
        else:
            self.snake.pop()       # no apple → remove tail

        # ── Max steps truncation ─────────────────────────────────────────────
        if self.steps >= self.max_steps:
            self.done = True
            return self._get_state(), reward, True

        return self._get_state(), reward, False

    # ── State representation ──────────────────────────────────────────────────

    def _get_state(self):
        """
        Build the 11-value feature vector the agent will learn from.

        Danger flags use the snake's current direction to define
        "straight", "right", "left" — so the agent reasons in relative
        terms, not absolute grid coordinates.
        """
        hx, hy = self.snake[0]

        # ── Relative directions from current movement ─────────────────────
        # straight = current direction
        # right    = 90° clockwise turn
        # left     = 90° counter-clockwise turn
        straight = (self.dx,       self.dy)
        right    = (-self.dy,      self.dx)
        left     = ( self.dy,     -self.dx)

        danger_straight = 1.0 if self._is_fatal((hx + straight[0], hy + straight[1])) else 0.0
        danger_right    = 1.0 if self._is_fatal((hx + right[0],    hy + right[1]))    else 0.0
        danger_left     = 1.0 if self._is_fatal((hx + left[0],     hy + left[1]))     else 0.0

        # ── Current direction (one-hot) ───────────────────────────────────
        dir_up    = 1.0 if (self.dx, self.dy) == ( 0, -1) else 0.0
        dir_down  = 1.0 if (self.dx, self.dy) == ( 0,  1) else 0.0
        dir_left  = 1.0 if (self.dx, self.dy) == (-1,  0) else 0.0
        dir_right = 1.0 if (self.dx, self.dy) == ( 1,  0) else 0.0

        # ── Nearest apple direction ───────────────────────────────────────
        ax, ay = self._nearest_apple()
        apple_up    = 1.0 if ay < hy else 0.0
        apple_down  = 1.0 if ay > hy else 0.0
        apple_left  = 1.0 if ax < hx else 0.0
        apple_right = 1.0 if ax > hx else 0.0

        return [
            danger_straight, danger_right, danger_left,
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
            return (self.grid_w // 2, self.grid_h // 2)  # fallback
        hx, hy = self.snake[0]
        return min(self.apples, key=lambda a: abs(a[0] - hx) + abs(a[1] - hy))

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
        Try to place a wall following Google Snake's placement rules:
          - Not on the snake
          - Not on an apple
          - At least 1-tile gap from existing walls (no adjacency)
          - At least 3 taxicab distance from head
          - Not in the 3×3 corners of the board
        """
        snake_set = set(self.snake)
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

            # 1-tile gap from existing walls
            too_close_to_wall = any(
                abs(wx - x) <= 1 and abs(wy - y) <= 1
                for wx, wy in wall_set
            )
            if too_close_to_wall:
                continue

            # Taxicab distance from head
            if abs(x - hx) + abs(y - hy) <= 3:
                continue

            # Corner protection (3×3 corners)
            corners = [
                (0, 0), (self.grid_w - 1, 0),
                (0, self.grid_h - 1), (self.grid_w - 1, self.grid_h - 1),
            ]
            if any(abs(x - cx) <= 1 and abs(y - cy) <= 1 for cx, cy in corners):
                continue

            self.walls.append((x, y))
            break  # success

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_grid(self):
        """
        Return the full 17×15 grid as a 2D list (for debugging / future CNN use).
        grid[y][x] is one of EMPTY, SNAKE_HEAD, SNAKE_BODY, APPLE, WALL.
        """
        grid = [[EMPTY] * self.grid_w for _ in range(self.grid_h)]

        for wx, wy in self.walls:
            grid[wy][wx] = WALL
        for ax, ay in self.apples:
            grid[ay][ax] = APPLE
        for i, (sx, sy) in enumerate(self.snake):
            grid[sy][sx] = SNAKE_HEAD if i == 0 else SNAKE_BODY

        return grid

    def print_grid(self):
        """ASCII render — useful for debugging without pygame."""
        symbols = {EMPTY: ".", SNAKE_HEAD: "H", SNAKE_BODY: "s", APPLE: "A", WALL: "#"}
        grid = self.get_grid()
        print("+" + "-" * self.grid_w + "+")
        for row in grid:
            print("|" + "".join(symbols[c] for c in row) + "|")
        print("+" + "-" * self.grid_w + "+")
        print(f"Score: {self.apples_eaten}  Step: {self.steps}  Walls: {len(self.walls)}")

    def render(self, cell_size=60, fps=10):
        """
        Draw the current game state in a pygame window.
        Call once per step. Handles its own clock.

        Args:
            cell_size : pixel size of each grid cell (default 60 matches the JS game)
            fps       : how fast the bot plays visually
        """
        import pygame

        # ── Init pygame on first call ─────────────────────────────────────────
        if not hasattr(self, '_screen'):
            pygame.init()
            w = self.grid_w * cell_size
            h = self.grid_h * cell_size + 40   # extra 40px for score bar
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Snake RL")
            self._clock  = pygame.time.Clock()
            self._font   = pygame.font.SysFont("Arial", 20)

        # ── Handle window close ───────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # ── Background (checkerboard) ─────────────────────────────────────────
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                color = (170, 215, 81) if (x + y) % 2 == 0 else (162, 209, 73)
                pygame.draw.rect(self._screen, color,
                                 (x * cell_size, y * cell_size, cell_size, cell_size))

        # ── Walls ─────────────────────────────────────────────────────────────
        for wx, wy in self.walls:
            pygame.draw.rect(self._screen, (83, 138, 52),
                             (wx * cell_size, wy * cell_size, cell_size, cell_size))
            # inner shadow to look like the JS game's bush
            pygame.draw.rect(self._screen, (60, 110, 35),
                             (wx * cell_size + 6, wy * cell_size + 6,
                              cell_size - 12, cell_size - 12))

        # ── Apples ────────────────────────────────────────────────────────────
        for ax, ay in self.apples:
            cx = ax * cell_size + cell_size // 2
            cy = ay * cell_size + cell_size // 2
            pygame.draw.circle(self._screen, (220, 50, 50), (cx, cy), cell_size // 2 - 4)

        # ── Snake ─────────────────────────────────────────────────────────────
        for i, (sx, sy) in enumerate(self.snake):
            color = (30, 90, 180) if i == 0 else (75, 133, 212)   # head darker
            pygame.draw.rect(self._screen, color,
                             (sx * cell_size + 2, sy * cell_size + 2,
                              cell_size - 4, cell_size - 4),
                             border_radius=8)

        # ── Score bar ─────────────────────────────────────────────────────────
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

    print(f"State length : {len(state)}  (expected 11)")
    print(f"State values : {state}")
    print()
    env.print_grid()

    # Play 10 random steps
    for i in range(10):
        action = random.randint(0, 3)
        state, reward, done = env.step(action)
        print(f"Step {i+1}: action={action}  reward={reward:.3f}  done={done}")
        if done:
            print("Episode ended.")
            break

    print("\nEnv is working correctly.")
