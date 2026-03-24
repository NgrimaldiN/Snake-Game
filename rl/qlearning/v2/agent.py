from snake_env import SnakeEnv 
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import os

episodes=20000
alpha=0.1
gamma=0.999
epsilon=1.0
epsilon_decay=0.01**(1/(episodes*0.75))
epsilon_min=0.01

env=SnakeEnv()
state=env.reset()

def state_to_index(state):
    index=0
    for bit in state:
        index=(index<<1)|int(bit)
    return index

n_states=2**13
n_actions=4

q_table=np.zeros((n_states,n_actions))

def choose_action(state):
    if rd.uniform(0,1)<epsilon:
        return rd.randint(0,3)
    else :
        return np.argmax(q_table[state_to_index(state),:])


scores    = []
steps_log = []
for episode in range(episodes):
    state=env.reset()
    done = False
    step = 0
    while not done :
        action=choose_action(state)
        next_state,reward,done=env.step(action)
        next_state_value=max(q_table[state_to_index(next_state),:])
        old_value=q_table[state_to_index(state),action]
        q_table[state_to_index(state),action]=(1-alpha)*old_value + alpha*( reward + gamma*next_state_value)
        state=next_state
        step += 1
    epsilon=max(epsilon*epsilon_decay,epsilon_min)
    scores.append(env.apples_eaten)
    steps_log.append(step)
    print(f"Episode {episode+1} | score: {env.apples_eaten} | steps: {step} | eps: {epsilon:.3f}")


# ── Plot progression ──────────────────────────────────────────────────────────

def smooth(data, window=50):
    return [sum(data[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(data))]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

smoothed_scores = smooth(scores)
convergence_val = np.mean(smoothed_scores[-int(episodes*0.1):])

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
save_path = os.path.join(os.path.dirname(__file__), "progression.png")
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




        



        