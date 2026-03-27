"""
Browser bot — plays Google Snake (googlesnakemods.com) using the trained DQN.

Reads game state directly from the game's internal JS objects (injected via
route interception) instead of pixel parsing — gives perfect state accuracy.

Usage:
    python3 play_browser.py                    # play with best_model.pt
    python3 play_browser.py --calibrate        # print grid from game internals
    python3 play_browser.py checkpoints/X.pt   # use a specific checkpoint
"""

import os
import sys
import re
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from playwright.sync_api import sync_playwright

# ── Grid (small map: 10 wide × 9 tall) ──────────────────────────────────────
GRID_W, GRID_H = 10, 9

# ── Actions — must match snake_env.py ────────────────────────────────────────
ACTION_KEYS = {0: "ArrowUp", 1: "ArrowDown", 2: "ArrowLeft", 3: "ArrowRight"}
DIR_CHANNEL = {(0, -1): 4, (0, 1): 5, (-1, 0): 6, (1, 0): 7}

DIR_FROM_STR = {
    "UP":    (0, -1),
    "DOWN":  (0,  1),
    "LEFT":  (-1, 0),
    "RIGHT": (1,  0),
    "NONE":  (1,  0),
}

GAME_URL = "https://googlesnakemods.com/v/current/"

# ── Route interception ──────────────────────────────────────────────────────
# The game class constructor contains `this.menu=X;this.header=X;` with
# readable (unminified) property names.  We inject `window.__game=this;`
# right after so we can read live game state from page.evaluate().

def _make_route_handler():
    """Return a Playwright route handler that injects the game hook."""
    injected = {"done": False}

    def handler(route):
        resp = route.fetch()
        body = resp.text()

        if not injected["done"] and "this.menu=" in body and "this.header=" in body:
            body, n = re.subn(
                r"(this\.menu=\w;this\.header=\w;)",
                r"\1window.__game=this;",
                body,
                count=1,
            )
            if n:
                injected["done"] = True
                print(f"  [hook] Injected window.__game into game source ({len(body)} bytes)")

        route.fulfill(response=resp, body=body)

    return handler


# ── JS state reader ─────────────────────────────────────────────────────────
# Runs inside the browser via page.evaluate().  Returns structured game state
# directly from the game's internal objects — zero pixel parsing.

JS_READ_STATE = """
() => {
    const g = window.__game;
    if (!g || !g.Aa || !g.Aa.oa || g.Aa.oa.length === 0)
        return {status: "no_game"};

    const dir = g.Aa.direction;
    if (dir === "NONE")
        return {status: "not_started"};
    if (g.Uj)
        return {status: "dead", score: g.Eh};

    const head = g.Aa.oa[0];
    if (head.x % 1 !== 0 || head.y % 1 !== 0)
        return {status: "animating"};

    return {
        status:    "alive",
        snake:     g.Aa.oa.map(p => [Math.round(p.x), Math.round(p.y)]),
        apples:    g.Ba.oa.map(f => [Math.round(f.pos.x), Math.round(f.pos.y)]),
        walls:     [],
        gridW:     g.oa.Aa.width,
        gridH:     g.oa.Aa.height,
        direction: dir,
        ticks:     g.ticks,
        score:     g.Eh
    };
}
"""


# ── State building ──────────────────────────────────────────────────────────

def build_state(data):
    """Convert JS game state dict -> 8-channel numpy array matching snake_env._get_state()."""
    gw, gh = data["gridW"], data["gridH"]
    state = np.zeros((8, gh, gw), dtype=np.uint8)

    hx, hy = data["snake"][0]
    if 0 <= hy < gh and 0 <= hx < gw:
        state[0, hy, hx] = 1

    for bx, by in data["snake"][1:]:
        if 0 <= by < gh and 0 <= bx < gw:
            state[1, by, bx] = 1

    for ax, ay in data["apples"]:
        if 0 <= ay < gh and 0 <= ax < gw:
            state[2, ay, ax] = 1

    dx, dy = DIR_FROM_STR.get(data["direction"], (1, 0))
    state[DIR_CHANNEL[(dx, dy)]] = 1

    return state


# ── DQN (same architecture as agent.py) ─────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, in_ch, h, w, n_act):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        flat = 64 * math.ceil(h / 2) * math.ceil(w / 2)
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, n_act)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    calibrate_only = "--calibrate" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    base_dir = os.path.dirname(__file__)
    ckpt_path = args[0] if args else os.path.join(base_dir, "checkpoints", "best_model.pt")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    net = None
    if not calibrate_only:
        if not os.path.exists(ckpt_path):
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        net = DQN(8, GRID_H, GRID_W, 4).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["policy_state_dict"])
        net.eval()
        print(f"Model: episode {ckpt.get('episode', '?')}, "
              f"best avg50: {ckpt.get('best_avg50', '?')}, device: {device}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.route("**/*.js", _make_route_handler())
        page.route("**/*.js?*", _make_route_handler())

        page.goto(GAME_URL, wait_until="networkidle")

        print("\n" + "=" * 60)
        print("  Configure the game on googlesnakemods.com:")
        print("    1. Settings  ->  5 apples, wall, small map")
        print("    2. Press Play, then start moving the snake")
        print("=" * 60)
        try:
            input("\nPress Enter when the game is running... ")
        except EOFError:
            wait = 40
            print(f"\nWaiting {wait}s for you to configure the game...")
            for i in range(wait, 0, -1):
                print(f"\r  Starting in {i}s...  ", end="", flush=True)
                time.sleep(1)
            print()

        # Verify hook was injected and game is accessible
        for attempt in range(30):
            result = page.evaluate(JS_READ_STATE)
            if result and result["status"] != "no_game":
                break
            time.sleep(0.5)
        else:
            sys.exit("Could not access game state — hook may not have been injected.")

        print(f"  Game state accessible!  status={result['status']}")

        # ── Calibrate mode ──────────────────────────────────────────────
        if calibrate_only:
            if result["status"] == "not_started":
                print("  Game not started yet — start moving the snake first.")
                while True:
                    result = page.evaluate(JS_READ_STATE)
                    if result["status"] == "alive":
                        break
                    time.sleep(0.3)

            if result["status"] != "alive":
                sys.exit(f"Unexpected game status for calibration: {result['status']}")

            gw, gh = result["gridW"], result["gridH"]
            print(f"\n  Grid: {gw} x {gh}  (expected {GRID_W} x {GRID_H})")
            print(f"  Snake length: {len(result['snake'])}")
            print(f"  Apples: {len(result['apples'])}")
            print(f"  Direction: {result['direction']}")
            print(f"  Score: {result.get('score', '?')}")

            state = build_state(result)
            print(f"\n  8-channel state grid (head={result['snake'][0]}):")
            for row in range(gh):
                cells = []
                for col in range(gw):
                    if   state[0, row, col]: cells.append("H")
                    elif state[1, row, col]: cells.append("S")
                    elif state[2, row, col]: cells.append("A")
                    elif state[3, row, col]: cells.append("W")
                    else:                    cells.append(".")
                print(f"    row {row}: {' '.join(cells)}")
            print(f"\n  Legend: H=head S=snake A=apple W=wall .=empty")

            dir_ch = None
            for d, ch in DIR_CHANNEL.items():
                if state[ch].any():
                    dir_ch = (d, ch)
            if dir_ch:
                print(f"  Direction channel: {dir_ch[1]} = {dir_ch[0]}")

            print("\nBrowser stays open — compare grid above with the game.")
            print("Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            browser.close()
            return

        # ── Bot loop ────────────────────────────────────────────────────
        print("\nBot running!  (Ctrl+C to stop)\n")
        tick = 0
        games = 0
        last_ticks = -1

        def restart_game():
            nonlocal games, last_ticks
            games += 1
            print(f"  Game over (game #{games}) — restarting...")
            time.sleep(0.8)
            page.keyboard.press("Enter")
            time.sleep(0.6)
            page.keyboard.press("Enter")
            last_ticks = -1

        try:
            while True:
                result = page.evaluate(JS_READ_STATE)
                status = result["status"]

                if status == "no_game":
                    time.sleep(0.5)
                    continue

                if status == "dead":
                    restart_game()
                    continue

                if status == "not_started":
                    time.sleep(0.2)
                    continue

                if status == "animating":
                    time.sleep(0.01)
                    continue

                game_ticks = result["ticks"]
                if game_ticks == last_ticks:
                    time.sleep(0.01)
                    continue
                last_ticks = game_ticks

                state = build_state(result)

                with torch.no_grad():
                    t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
                    action = net(t).argmax(dim=1).item()

                page.keyboard.press(ACTION_KEYS[action])
                tick += 1

                if tick % 50 == 0:
                    head = result["snake"][0]
                    score = result.get("score", "?")
                    print(f"  tick {tick:>5}  |  game: {games+1}  |  head: {head}  "
                          f"|  score: {score}  |  action: {ACTION_KEYS[action]}")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print(f"\nStopped after {tick} ticks, {games} restarts.")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
