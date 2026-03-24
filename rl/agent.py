from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT
import numpy as np
import random as rd

eps=1.0
eps_dec=0.998

a=0.1
gamma=0.99

num_episodes=1000
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
    print(f"Episode {episode+1} | steps: {step+1} | eps: {eps:.3f}")
    eps=max(eps*eps_dec,0.01)

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




        

