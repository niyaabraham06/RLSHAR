import gymnasium as gym
import numpy as np
import panda_gym
from agents.td3 import TD3Agent
import os

# 1. Initialize the Environment
# --- CHANGE: Use 'dense' rewards (Distance based) ---
# This gives the robot feedback every single step ("Hotter/Colder")
env = gym.make('PandaReach-v3', render_mode="rgb_array", reward_type="dense")

# 2. Calculate input size
obs_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]
combined_input_dims = (obs_dim + 2 * goal_dim,) 

# 3. Training Parameters
n_episodes = 1000   # Dense rewards solve this VERY fast (usually <500)
max_steps = 50     

# 4. Initialize Agent
agent = TD3Agent(
    alpha=0.001, beta=0.001, # We can use a faster learning rate for Dense rewards
    tau=0.005,
    env=env,
    input_dims=combined_input_dims,
    n_actions=env.action_space.shape[0],
    layer1_size=256, layer2_size=256,
    batch_size=256
)

best_score = float('-inf')
score_history = []

print("Starting Training (Dense Rewards Mode)...")

for i in range(n_episodes):
    observation, info = env.reset()
    done = False
    score = 0
    
    for step in range(max_steps):
        # Construct state
        state = np.concatenate([
            observation['observation'], 
            observation['achieved_goal'], 
            observation['desired_goal']
        ])
        
        # Choose Action
        action = agent.choose_action(state)
        
        # Step
        next_observation, reward, done, truncated, info = env.step(action)
        
        # Construct next state
        next_state = np.concatenate([
            next_observation['observation'], 
            next_observation['achieved_goal'], 
            next_observation['desired_goal']
        ])
        
        done = done or truncated
        
        # Remember & Learn
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        
        score += reward
        observation = next_observation
        
        if done:
            break

    # Logging
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    # --- SUCCESS CHECK FOR DENSE REWARDS ---
    # In Dense mode, a score close to 0 (e.g., > -5) is excellent.
    # It means the robot spent most of the time VERY close to the target.
    if avg_score > best_score and i > 20:
        best_score = avg_score
        agent.save_models()
        print(f"--- New Best Score: {avg_score:.2f} --- Model Saved!")

    print(f"Episode {i} | Score: {score:.2f} | Avg Score: {avg_score:.2f}")

env.close()