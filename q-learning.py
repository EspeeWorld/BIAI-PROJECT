import numpy as np
import random

# Environment setup
state_space = np.arange(-10, 11, 1)  # States from -10 to 10
action_space = [-1, 1]  # Actions: move left (-1) or right (+1)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000  # Number of episodes to train

# Q-table initialization
Q = np.zeros((len(state_space), len(action_space)))  # Initialize Q-table

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Function to get the index of the state in the state space
def get_state_index(state):
    return int(state + 10)  # Mapping state -10 to index 0, 10 to index 20

# Safe RL function
def safe_rl_agent(state):
    # Choose action based on epsilon-greedy policy
    if random.uniform(0, 1) < epsilon:
        action = random.choice(action_space)  # Explore
    else:
        state_idx = get_state_index(state)
        action = action_space[np.argmax(Q[state_idx])]  # Exploit
    return action

# Environment dynamics with updated safety rule
def step(state, action):
    new_state = state + action
    if new_state < -10:
        new_state = -10
    elif new_state > 10:
        new_state = 10

    # Updated reward structure:
    # High penalty outside -8 to 8
    if new_state < -8 or new_state > 8:
        reward = -10  # Danger zone penalty
    else:
        reward = 1  # Safe move

    if new_state == 10:
        reward = 5  # Goal reward overrides other rewards

    return new_state, reward

# Q-Learning with updated safety dynamics
def q_learning():
    for episode in range(episodes):
        state = random.choice(state_space)  # Start from a random state
        done = False

        while not done:
            action = safe_rl_agent(state)  # Choose action
            new_state, reward = step(state, action)  # Environment response

            state_idx = get_state_index(state)
            new_state_idx = get_state_index(new_state)
            action_idx = action_space.index(action)

            # Q-learning update
            Q[state_idx, action_idx] += alpha * (
                reward + gamma * np.max(Q[new_state_idx]) - Q[state_idx, action_idx]
            )

            state = new_state

            if state == 10:
                done = True

# Test the trained agent
def test_agent():
    state = -10  # Start at far left
    trajectory = []

    while state != 10:
        action = safe_rl_agent(state)
        trajectory.append((state, action))
        state, _ = step(state, action)

    trajectory.append((state, "Goal Reached"))
    return trajectory

# Train and test
q_learning()
trajectory = test_agent()

# Output
print("Agent's Path and Actions:")
for state, action in trajectory:
    print(f"State: {state}, Action: {action}")
