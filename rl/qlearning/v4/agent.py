### Q-Learning Agent v4 — 22-bit state, dict-based Q-table
import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import defaultdict
from snake_env import SnakeEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────
episodes      = 100000
alpha         = 0.1
gamma         = 0.99
epsilon       = 1.0
epsilon_min   = 0.01
epsilon_decay = epsilon_min ** (1 / (episodes * 0.70))

# ── Environment ───────────────────────────────────────────────────────────────
env       = SnakeEnv()
n_actions = 4

# ── Dict-based Q-table (sparse — only visited states stored) ──────────────────
# Returns array of zeros for unseen states
q_table = defaultdict(lambda: np.zeros(n_actions))


def state_to_key(state):
    """Convert state list to hashable tuple for dict lookup."""
    return tuple(int(b) for b in state)


def choose_action(state):
    if rd.random() < epsilon:
        return rd.randint(0, 3)
    else:
        return int(np.argmax(q_table[state_to_key(state)]))


# ── Training loop ─────────────────────────────────────────────────────────────
scores    = []
steps_log = []

for episode in range(episodes):
    state = env.reset()
    done  = False
    step  = 0

    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        s_key  = state_to_key(state)
        ns_key = state_to_key(next_state)

        old_value        = q_table[s_key][action]
        next_state_value = np.max(q_table[ns_key])
        q_table[s_key][action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_state_value)

        state = next_state
        step += 1

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    scores.append(env.apples_eaten)
    steps_log.append(step)
    if (episode + 1) % 500 == 0:
        avg_50 = np.mean(scores[-50:])
        print(f"Episode {episode+1:>6} | score: {env.apples_eaten:>3} | avg50: {avg_50:.1f} | "
              f"steps: {step:>4} | eps: {epsilon:.3f} | Q-states: {len(q_table)}")

print(f"\nTraining done. Unique Q-states visited: {len(q_table)}")

# ── Plot progression ──────────────────────────────────────────────────────────

def smooth(data, window=200):
    return [sum(data[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(data))]


fig, axes = plt.subplots(2, 1, figsize=(12, 8))

smoothed_scores = smooth(scores)
convergence_val = np.mean(smoothed_scores[-int(episodes * 0.1):])

axes[0].plot(scores, alpha=0.15, color="steelblue", label="raw")
axes[0].plot(smoothed_scores, color="steelblue", linewidth=2, label="smoothed (200 ep)")
axes[0].axhline(y=convergence_val, color='r', linestyle='--', label=f'Convergence (~{convergence_val:.1f})')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Score (apples eaten)")
axes[0].set_title("Score per episode — v4 (22-bit state)")
axes[0].legend()

axes[1].plot(steps_log, alpha=0.15, color="orange", label="raw")
axes[1].plot(smooth(steps_log), color="orange", linewidth=2, label="smoothed (200 ep)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Steps survived")
axes[1].set_title("Steps per episode")
axes[1].legend()

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), "progression.png")
plt.savefig(save_path)
plt.show()
print(f"Plot saved to {save_path}")

# ── Watch the trained agent play ──────────────────────────────────────────────
print("\nWatching agent play (close window to exit)...")

watch_env = SnakeEnv()
state     = watch_env.reset()

while True:
    watch_env.render(fps=8)
    action = int(np.argmax(q_table[state_to_key(state)]))
    state, _, done = watch_env.step(action)
    if done:
        state = watch_env.reset()
