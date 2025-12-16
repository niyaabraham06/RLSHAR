import gymnasium as gym
import numpy as np
import panda_gym
from agents.td3 import TD3Agent
import time

def main_demo():
    # 1. SETUP
    env = gym.make('PandaReach-v3', render_mode="human", reward_type="dense")

    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    combined_input_dims = (obs_dim + 2 * goal_dim,)

    # 2. LOAD AGENT
    agent = TD3Agent(
        alpha=0.001, beta=0.001, tau=0.005,
        env=env, input_dims=combined_input_dims,
        n_actions=env.action_space.shape[0],
        layer1_size=256, layer2_size=256,
        batch_size=256
    )
    
    print("Loading model...")
    agent.load_models()

    print("\n--- LIVE DEMO ---")
    
    episode = 1
    while True:
        observation, info = env.reset()
        done = False
        
        # Max 50 steps to hit the target
        for step in range(50):
            state = np.concatenate([
                observation['observation'], 
                observation['achieved_goal'], 
                observation['desired_goal']
            ])
            
            # evaluate=True is CRITICAL
            action = agent.choose_action(state, evaluate=True)
            
            # FAST EXECUTION (No sleep)
            observation, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                break

        # Check result
        dist_cm = np.linalg.norm(observation['achieved_goal'] - observation['desired_goal']) * 100
        
        if dist_cm < 5.0:
            print(f"Episode {episode} | ✅ HIT! ({dist_cm:.1f} cm)")
        else:
            print(f"Episode {episode} | ❌ Miss ({dist_cm:.1f} cm)")
        
        # Small pause only BETWEEN episodes
        time.sleep(0.5) 
        episode += 1

    env.close()

if __name__ == '__main__':
    main_demo()