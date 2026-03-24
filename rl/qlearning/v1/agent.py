###agent on 11 parameter env
import os

from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT
import numpy as np
import random as rd
import matplotlib.pyplot as plt
num_episodes=100000
eps=1.0
eps_dec=0.01**(1/(num_episodes*0.6))

a=0.1
gamma=0.99


max_steps=10000

env=SnakeEnv()
state=env.reset()
n_states=2**11
n_actions=4


def state_to_index(state):
    index = 0
    for bit in state:
        index = (index << 1) | int(bit)
    return index

q_table=np.zeros((n_states,n_actions))


def choose_action(state):
    if rd.uniform(0,1)<eps:
        return rd.randint(0,n_actions-1)
    else:
        return np.argmax(q_table[state_to_index(state),:])

scores    = []
steps_log = []

for episode in range(num_episodes):
    state=env.reset()
    for step in range(max_steps):
        action=choose_action(state)
        next_state,reward,done=env.step(action)

        old_value=q_table[state_to_index(state),action]
        next_max=np.max(q_table[state_to_index(next_state),:])
        q_table[state_to_index(state),action]=(1-a)*old_value+ a*(reward + gamma*next_max)
        state=next_state
        if done:
            break

    scores.append(env.apples_eaten)
    steps_log.append(step + 1)
    print(f"Episode {episode+1} | score: {env.apples_eaten} | steps: {step+1} | eps: {eps:.3f}")
    eps=max(eps*eps_dec,0.01)

# ── Plot progression ──────────────────────────────────────────────────────────

def smooth(data, window=50):
    return [sum(data[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(data))]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

smoothed_scores = smooth(scores)
convergence_val = np.mean(smoothed_scores[-int(num_episodes*0.1):])

axes[0].plot(scores, alpha=0.3, color="steelblue", label="raw")
axes[0].plot(smoothed_scores, color="steelblue", linewidth=2, label="smoothed (50 ep)")
axes[0].axhline(y=convergence_val, color='r', linestyle='--', label=f'Convergence (~{convergence_val:.1f})')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Score (apples eaten)")
axes[0].set_title("Score per episode")
axes[0].legend()

axes[1].plot(steps_log, alpha=0.3, color="orange", label="raw")
axes[1].plot(smooth(steps_log), color="orange", linewidth=2, label="smoothed (50 ep)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Steps survived")
axes[1].set_title("Steps per episode")
axes[1].legend()

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), "progression_optimized_decay.png")
plt.savefig(save_path)
plt.show()
print(f"Plot saved to {save_path}")

# ── Watch the trained agent play ──────────────────────────────────────────────
print("\nTraining done. Watching agent play (close the window to exit)...")

watch_env = SnakeEnv()
state     = watch_env.reset()

while True:
    watch_env.render(fps=8)
    action     = np.argmax(q_table[state_to_index(state), :])   # pure exploit, no random
    state, _, done = watch_env.step(action)
    if done:
        state = watch_env.reset()




        

