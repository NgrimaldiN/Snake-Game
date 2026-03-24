"""
Play Snake in the terminal to verify the Python env behaves correctly.
Controls: W/A/S/D or arrow keys. Q to quit.

Run with:
    python play_terminal.py
"""

import sys
import time
import os

# Windows keyboard input (no external library needed)
if sys.platform == "win32":
    import msvcrt
    def get_key():
        """Return a key press without waiting (non-blocking). None if no key."""
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):  # special key prefix (arrows)
                ch2 = msvcrt.getwch()
                return {
                    'H': 'UP', 'P': 'DOWN', 'K': 'LEFT', 'M': 'RIGHT'
                }.get(ch2)
            return ch.upper()
        return None
else:
    import tty, termios
    def get_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(2)
                return {'[A': 'UP', '[B': 'DOWN', '[C': 'RIGHT', '[D': 'LEFT'}.get(ch2)
            return ch.upper()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT

KEY_TO_ACTION = {
    'W': UP,    'UP':    UP,
    'S': DOWN,  'DOWN':  DOWN,
    'A': LEFT,  'LEFT':  LEFT,
    'D': RIGHT, 'RIGHT': RIGHT,
}

def clear():
    os.system('cls' if sys.platform == 'win32' else 'clear')

def main():
    env = SnakeEnv()
    state = env.reset()
    action = RIGHT  # default direction (snake faces right)

    # ── Frozen first frame — wait for first keypress before starting ──────────
    clear()
    env.print_grid()
    print("\nPress a direction key to start (W=Up / S=Down / D=Right) | Q to quit")
    while True:
        key = get_key()
        if key == 'Q':
            sys.exit()
        # Only allow forward (D/Right), up (W), down (S) — not left (reverse)
        if key in ('W', 'UP', 'S', 'DOWN', 'D', 'RIGHT'):
            action = KEY_TO_ACTION[key]
            break
        time.sleep(0.01)

    while True:
        clear()
        env.print_grid()
        print(f"\nState: {[round(s, 2) for s in state]}")
        print("W=Up  S=Down  A=Left  D=Right  Q=Quit")

        # Read key (give player 200ms to press something)
        key = None
        deadline = time.time() + 0.2
        while time.time() < deadline:
            key = get_key()
            if key:
                break
            time.sleep(0.01)

        if key == 'Q':
            print("Quit.")
            break

        if key in KEY_TO_ACTION:
            action = KEY_TO_ACTION[key]

        state, reward, done = env.step(action)

        if reward == 1.0:
            print(f"Apple eaten! Score: {env.apples_eaten}")
            time.sleep(0.1)

        if done:
            clear()
            env.print_grid()
            print(f"\nGame over! Final score: {env.apples_eaten}")
            print("Press any key to play again, Q to quit.")
            while True:
                key = get_key()
                if key == 'Q':
                    sys.exit()
                if key:
                    state = env.reset()
                    action = RIGHT
                    break
                time.sleep(0.01)

if __name__ == "__main__":
    main()
