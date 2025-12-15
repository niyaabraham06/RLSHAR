import gymnasium as gym
import numpy as np
import panda_gym
from agents.td3 import TD3Agent
import time

def smooth_demo():
    # 1. Setup Environment with Dense Rewards (Important!)
    env = gym.make('PandaReach-v3', render_mode="human", reward_type="dense")

    # 2. Dimensions
    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    combined_input_dims = (obs_dim + 2 * goal_dim,)

    # 3. Initialize Agent
    agent = TD3Agent(
        alpha=0.001, beta=0.001, tau=0.005,
        env=env, input_dims=combined_input_dims,
        n_actions=env.action_space.shape[0],
        layer1_size=256, layer2_size=256,
        batch_size=256
    )

    # 4. Load the Best Brain
    print("Loading model...")
    agent.load_models()

    print("\n--- STARTING SMOOTH DEMO ---")
    
    # Run 5 demo episodes
    for i in range(5): 
        observation, info = env.reset()
        done = False
        
        # --- SMOOTHING SETUP ---
        # We start with a neutral action
        previous_action = np.zeros(env.action_space.shape[0])
        
        # FACTOR: 0.2 means "Keep 20% of old action, use 80% of new action"
        # This is very responsive but cuts out the tiny "jitters".
        smoothing_factor = 0.2 
        
        print(f"\nDemo {i+1} Started...")
        
        while not done:
            state = np.concatenate([
                observation['observation'], 
                observation['achieved_goal'], 
                observation['desired_goal']
            ])
            
            # 1. Get the raw action from the AI
            raw_action = agent.choose_action(state, evaluate=True)
            
            # 2. Apply Smoothing Formula
            smooth_action = (smoothing_factor * previous_action) + ((1 - smoothing_factor) * raw_action)
            
            # 3. Send smooth action to robot
            observation, reward, done, truncated, info = env.step(smooth_action)
            
            # Update previous action
            previous_action = smooth_action
            
            done = done or truncated
            
            # 4. Small delay to make movement look natural to the eye
            time.sleep(0.02) 

        # Visual Feedback
        dist = np.linalg.norm(observation['achieved_goal'] - observation['desired_goal'])
        if dist < 0.05: 
            print(f"Result: HIT! (Distance: {dist*100:.1f} cm)")
        else:
            print(f"Result: Miss (Distance: {dist*100:.1f} cm)")
            
        time.sleep(1.0) 

    env.close()

if __name__ == '__main__':
    smooth_demo()