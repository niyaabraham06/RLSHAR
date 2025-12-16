import gymnasium as gym
import numpy as np
import panda_gym
from agents.td3 import TD3Agent
import os
import shutil

def emergency_retrain():
    # 1. CLEANUP (Do this first)
    if os.path.exists('ckp/TD3_HER_Panda'):
        try:
            shutil.rmtree('ckp/TD3_HER_Panda')
        except:
            pass

    # 2. SETUP
    env = gym.make('PandaReach-v3', render_mode="rgb_array", reward_type="dense")

    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    combined_input_dims = (obs_dim + 2 * goal_dim,) 

    # 3. FAST TRAINING (200 episodes is enough)
    n_episodes = 200  
    max_steps = 50     

    agent = TD3Agent(
        alpha=0.001, beta=0.001, tau=0.005,
        env=env, input_dims=combined_input_dims,
        n_actions=env.action_space.shape[0],
        layer1_size=256, layer2_size=256,
        batch_size=256
    )
    
    best_score = float('-inf')
    score_history = []
    
    print("--- STARTING FINAL RETRAINING ---")
    print("Target: Save WHENEVER the score improves.")

    for i in range(n_episodes):
        observation, info = env.reset()
        done = False
        score = 0
        
        for step in range(max_steps):
            state = np.concatenate([
                observation['observation'], 
                observation['achieved_goal'], 
                observation['desired_goal']
            ])
            
            action = agent.choose_action(state)
            next_observation, reward, done, truncated, info = env.step(action)
            
            next_state = np.concatenate([
                next_observation['observation'], 
                next_observation['achieved_goal'], 
                next_observation['desired_goal']
            ])
            
            done = done or truncated
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            score += reward
            observation = next_observation
            
            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # --- GUARANTEED SAVE ---
        # If the average improves, SAVE. No other conditions.
        # We wait 5 episodes just to let the buffer fill up slightly.
        if avg_score > best_score and i > 5:
            best_score = avg_score
            agent.save_models()
            print(f"!!! NEW WINNING MODEL SAVED (Score: {avg_score:.2f}) !!!")

        if i % 10 == 0:
            print(f"Episode {i} | Score: {score:.2f} | Avg: {avg_score:.2f}")

    print("Training Finished.")
    env.close()

if __name__ == '__main__':
    emergency_retrain()