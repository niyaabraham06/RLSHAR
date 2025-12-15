import gymnasium as gym
import numpy as np
import panda_gym
from agents.td3 import TD3Agent
import time

def main_demo():
    # 1. SETUP: Use 'dense' rewards. This is CRITICAL.
    # If you use 'sparse', the robot will fail.
    env = gym.make('PandaReach-v3', render_mode="human", reward_type="dense")

    # 2. DIMENSIONS
    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    combined_input_dims = (obs_dim + 2 * goal_dim,)

    # 3. LOAD AGENT
    agent = TD3Agent(
        alpha=0.001, beta=0.001, tau=0.005,
        env=env, input_dims=combined_input_dims,
        n_actions=env.action_space.shape[0],
        layer1_size=256, layer2_size=256,
        batch_size=256
    )
    
    # Load the trained weights
    print("Loading the winning model...")
    agent.load_models()

    print("\n--- RUNNING FINAL DEMO ---")
    print("Robot should reach the green ball.")

    episode = 1
    while True:
        observation, info = env.reset()
        done = False
        score = 0
        
        print(f"Episode {episode} started...")
        
        while not done:
            state = np.concatenate([
                observation['observation'], 
                observation['achieved_goal'], 
                observation['desired_goal']
            ])
            
            # evaluate=True is MANDATORY. 
            # It turns off training noise so the robot acts precisely.
            action = agent.choose_action(state, evaluate=True)
            
            # Execute action
            observation, reward, done, truncated, info = env.step(action)
            
            # VISUAL SMOOTHING TRICK
            # We pause for 0.04 seconds. This makes the movement look fluid
            # to your eyes, but doesn't mess up the robot's math.
            time.sleep(0.04) 
            
            score += reward
            done = done or truncated

        # Check result
        # In Dense mode, getting closer than 5cm is a win.
        dist = np.linalg.norm(observation['achieved_goal'] - observation['desired_goal'])
        dist_cm = dist * 100
        
        if dist_cm < 5.0:
            print(f"✅ HIT! Distance: {dist_cm:.1f} cm")
            time.sleep(1.0) # Pause to celebrate
        else:
            print(f"❌ Miss. Distance: {dist_cm:.1f} cm")
            time.sleep(0.5)
            
        episode += 1

    env.close()

if __name__ == '__main__':
    main_demo()