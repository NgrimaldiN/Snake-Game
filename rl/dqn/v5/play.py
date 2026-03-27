"""Load a saved checkpoint and watch the agent play Snake."""

import os
import sys
import math
import numpy as np
import torch

from snake_env import SnakeEnv

# ── Same network definition as agent.py ──────────────────────────────────────

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels, h, w, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        conv_h = math.ceil(h / 2)
        conv_w = math.ceil(w / 2)
        flat_size = 64 * conv_h * conv_w

        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── Load checkpoint ──────────────────────────────────────────────────────────

base_dir = os.path.dirname(__file__)
default_path = os.path.join(base_dir, "checkpoints", "best_model.pt")
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else default_path

if not os.path.exists(ckpt_path):
    sys.exit(f"Checkpoint not found: {ckpt_path}")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

env = SnakeEnv()
state_shape = env.reset().shape
n_actions = 4

net = DQN(state_shape[0], state_shape[1], state_shape[2], n_actions).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
net.load_state_dict(ckpt["policy_state_dict"])
net.eval()

ep = ckpt.get("episode", "?")
best = ckpt.get("best_avg50", "?")
print(f"Loaded checkpoint: {ckpt_path}")
print(f"  trained for {ep} episodes  |  best avg50: {best}")
print(f"  device: {device}")
print("Close the window to quit.\n")

# ── Play loop ────────────────────────────────────────────────────────────────

state = env.reset()
games = 0
total_score = 0

while True:
    env.render(fps=10, title=f"DQN v5  |  Game {games + 1}  |  Score: {env.apples_eaten}")
    with torch.no_grad():
        t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        action = net(t).argmax(dim=1).item()
    state, _, done = env.step(action)
    if done:
        games += 1
        total_score += env.apples_eaten
        print(f"Game {games:>3}  score: {env.apples_eaten:>3}   avg: {total_score/games:.1f}")
        state = env.reset()
