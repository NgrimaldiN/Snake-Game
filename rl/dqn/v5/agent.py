"""
DQN Agent v5 — CNN on raw 8-channel grid state.

4 new concepts over tabular Q-learning (v4):
  1. Neural network    — replaces the Q-table dict
  2. Replay buffer     — stores past transitions, trains on random mini-batches
  3. Target network    — frozen copy of the net, updated periodically (stabilises learning)
  4. Gradient descent  — replaces the direct Q[s][a] = ... update
"""

import os
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from snake_env import SnakeEnv

RESUME = "--resume" in sys.argv

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPISODES        = 250_000
BATCH_SIZE      = 128
GAMMA           = 0.99
LR              = 5e-5          # halved: stabilise near-optimal policy
BUFFER_CAPACITY = 100_000
TAU             = 0.005         # soft target update: blend 0.5% per train step
TRAIN_EVERY     = 4             # train once every 4 env steps (like Atari DQN)
EPS_START       = 1.0
EPS_END         = 0.01
EPS_DECAY_UNTIL = 70_000       # fixed: eps already at min, don't restart decay

CHECKPOINT_EVERY = 5_000
BEST_WINDOW      = 50
ENABLE_TRAINING_DEMOS = False

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REPLAY BUFFER
#    Stores (s, a, r, s', done) transitions in pre-allocated numpy arrays.
#    Sampling a random mini-batch breaks temporal correlation → stable training.
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.pos  = 0
        self.size = 0
        # uint8 because every channel value is 0 or 1 — saves 4× memory vs float32
        self.states      = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = done
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.from_numpy(self.states[idx].astype(np.float32)).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx].astype(np.float32)).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DQN NETWORK (CNN)
#    Takes the 8-channel grid and outputs one Q-value per action.
#
#    Architecture:
#      Conv 3×3 (8→32)  ──ReLU──▶  keeps 9×10
#      Conv 3×3 (32→64, stride 2) ──ReLU──▶  shrinks to 5×5
#      Conv 3×3 (64→64) ──ReLU──▶  keeps 5×5
#      Flatten (1600) ──▶ FC 256 ──ReLU──▶ FC 4  (Q-values)
# ═══════════════════════════════════════════════════════════════════════════════

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
        x = x.flatten(1)           # (batch, 1600)
        x = F.relu(self.fc1(x))
        return self.fc2(x)         # (batch, 4) — raw Q-values, no activation


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SETUP
# ═══════════════════════════════════════════════════════════════════════════════

env         = SnakeEnv()
n_actions   = 4
state_shape = env.reset().shape        # (8, 9, 10)
base_dir    = os.path.dirname(__file__)

# Policy network — the one we actually train
policy_net = DQN(state_shape[0], state_shape[1], state_shape[2], n_actions).to(device)

# Target network — soft-updated every step via Polyak averaging (TAU)
target_net = DQN(state_shape[0], state_shape[1], state_shape[2], n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()                       # never call .train() on this

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer    = ReplayBuffer(BUFFER_CAPACITY, state_shape)

param_count = sum(p.numel() for p in policy_net.parameters())
print(f"Network: {state_shape} -> {n_actions} actions  ({param_count:,} parameters)")

checkpoints_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

start_episode = 0
resumed_scores = []

if RESUME:
    ckpt_path = os.path.join(checkpoints_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"No checkpoint found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    target_net.load_state_dict(ckpt["target_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_episode = ckpt["episode"]
    resumed_scores = ckpt.get("scores", [])
    print(f"Resumed from episode {start_episode:,}  (best avg50: {ckpt['best_avg50']:.1f})")

def save_checkpoint(path, episode, best_avg50):
    torch.save(
        {
            "episode": episode,
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_avg50": best_avg50,
            "total_steps": total_steps,
            "scores": scores,
        },
        path,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ACTION SELECTION  (same ε-greedy as tabular, just uses the net instead)
# ═══════════════════════════════════════════════════════════════════════════════

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    with torch.no_grad():
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        return policy_net(state_t).argmax(dim=1).item()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OPTIMISATION STEP  (the core DQN learning update)
#
#    Tabular:   Q[s][a] = (1-α)·Q[s][a] + α·(r + γ·max Q[s'])
#    DQN:       loss = Huber( Q_policy(s)[a]  ,  r + γ·max Q_target(s') )
#               then backprop + gradient step
# ═══════════════════════════════════════════════════════════════════════════════

def optimise():
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    # Q_policy(s, a) — predicted Q for the action we actually took
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN target: policy net picks the best action, target net rates it
    with torch.no_grad():
        best_actions = policy_net(next_states).argmax(dim=1)
        next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + GAMMA * next_q * (1.0 - dones)

    loss = F.smooth_l1_loss(q_values, targets)   # Huber loss — less sensitive to outliers

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEMO FUNCTION  (plays one greedy game with rendering at milestone episodes)
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_AT = {1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000,
           10_000, 25_000, 50_000, 75_000, EPISODES}

demo_env = SnakeEnv()

def play_demo(episode_num):
    """Run one full game using greedy policy and render it visually."""
    import time
    state = demo_env.reset()
    done  = False
    title = f"Demo @ episode {episode_num:,}  (close window = skip)"

    while not done:
        demo_env.render(fps=12, title=title)
        with torch.no_grad():
            t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            action = policy_net(t).argmax(dim=1).item()
        state, _, done = demo_env.step(action)

    # Show final frame for a moment
    demo_env.render(fps=1, title=f"{title}  |  SCORE: {demo_env.apples_eaten}")
    time.sleep(1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

scores      = resumed_scores
steps_log   = []
loss_log    = []
total_steps = 0
best_avg50  = float("-inf")

for episode in range(start_episode, EPISODES):
    state = env.reset()
    done  = False
    step  = 0
    ep_loss = 0.0

    # Linear epsilon decay (simpler than exponential, standard in DQN papers)
    epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * episode / EPS_DECAY_UNTIL)

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)

        buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        step += 1
        total_steps += 1

        if total_steps % TRAIN_EVERY == 0 and len(buffer) >= BATCH_SIZE:
            ep_loss += optimise()
            for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                tp.data.mul_(1 - TAU).add_(pp.data, alpha=TAU)

    scores.append(env.apples_eaten)
    steps_log.append(step)
    loss_log.append(ep_loss / max(step, 1))

    if (episode + 1) % 500 == 0:
        avg = np.mean(scores[-50:])
        best_in_batch = max(scores[-500:])
        print(f"Ep {episode+1:>6} | best500 {best_in_batch:>3} | avg50 {avg:.1f} | "
              f"steps {step:>4} | eps {epsilon:.3f} | loss {np.mean(loss_log[-50:]):.4f} | "
              f"buf {len(buffer):,}")

    if ENABLE_TRAINING_DEMOS and (episode + 1) in DEMO_AT:
        print(f"  >>> Demo at episode {episode+1:,} ...")
        play_demo(episode + 1)

    if len(scores) >= BEST_WINDOW:
        avg50 = float(np.mean(scores[-BEST_WINDOW:]))
        if avg50 > best_avg50:
            best_avg50 = avg50
            best_path = os.path.join(checkpoints_dir, "best_model.pt")
            save_checkpoint(best_path, episode + 1, best_avg50)

    if (episode + 1) % CHECKPOINT_EVERY == 0:
        ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_ep_{episode+1:06d}.pt")
        save_checkpoint(ckpt_path, episode + 1, best_avg50)

print(f"\nDone. Total steps: {total_steps:,}")
last_path = os.path.join(checkpoints_dir, "last_model.pt")
save_checkpoint(last_path, EPISODES, best_avg50)
print(f"Saved final checkpoint to {last_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def smooth(data, window=200):
    return [sum(data[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(data))]


fig, axes = plt.subplots(3, 1, figsize=(12, 10))

smoothed_scores = smooth(scores)
convergence_val = np.mean(smoothed_scores[-int(EPISODES * 0.1):])

axes[0].plot(scores, alpha=0.15, color="steelblue", label="raw")
axes[0].plot(smoothed_scores, color="steelblue", linewidth=2, label="smoothed (200 ep)")
axes[0].axhline(y=convergence_val, color="r", linestyle="--",
                label=f"convergence (~{convergence_val:.1f})")
axes[0].set_ylabel("Score")
axes[0].set_title("Score per episode — v5 DQN (CNN on raw grid)")
axes[0].legend()

axes[1].plot(steps_log, alpha=0.15, color="orange", label="raw")
axes[1].plot(smooth(steps_log), color="orange", linewidth=2, label="smoothed")
axes[1].set_ylabel("Steps survived")
axes[1].set_title("Steps per episode")
axes[1].legend()

axes[2].plot(smooth(loss_log, 500), color="crimson", linewidth=1.5)
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Avg loss")
axes[2].set_title("Training loss (smoothed 500 ep)")

plt.tight_layout()
save_path = os.path.join(base_dir, "progression.png")
plt.savefig(save_path)
plt.show()
print(f"Plot saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. WATCH THE TRAINED AGENT PLAY
# ═══════════════════════════════════════════════════════════════════════════════

print("\nWatching agent play (close window to exit)...")

watch_env = SnakeEnv()
state = watch_env.reset()

while True:
    watch_env.render(fps=8)
    with torch.no_grad():
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        action = policy_net(state_t).argmax(dim=1).item()
    state, _, done = watch_env.step(action)
    if done:
        state = watch_env.reset()
