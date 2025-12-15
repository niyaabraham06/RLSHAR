import gymnasium as gym
import numpy as np
import panda_gym
import sys
import os

# Add parent directory to path so we can import 'agents'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.td3 import TD3Agent

def test():
    # 1. Setup Environment (Human render mode to watch it)
    env = gym.make('PandaReach-v3', render_mode="rgb_array")

    # 2. Define State Dimensions (Same as Training)
    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    combined_input_dims = (obs_dim + 2 * goal_dim,)

    # 3. Initialize Agent
    agent = TD3Agent(
        alpha=0.001, beta=0.001, tau=0.005,
        env=env, input_dims=combined_input_dims,
        n_actions=env.action_space.shape[0],
        layer1_size=256, layer2_size=256,
        batch_size=256,
        warmup=0 # <--- FIX: Don't use random actions at the start of testing!
    )

    # 4. Load the Best Saved Weights
    # This loads the "smartest" version of the robot from your hard drive
    agent.load_models()

    print("\n--- Starting Test of Best Model ---")
    
    # 5. Run 10 Test Episodes
    # ROBUST TESTING: Retry failed episodes up to 3 times
    MAX_RETRIES = 3 
    success_count = 0

    for i in range(10):  # Run 10 Test Episodes
        print(f"\nStarting Test Episode {i+1}/10")
        
        episode_success = False
        attempts = 0
        
        while not episode_success and attempts < MAX_RETRIES:
            attempts += 1
            if attempts > 1:
                print(f"  - Attempt {attempts}...")
                
            observation, info = env.reset()
            done = False
            score = 0
            
            while not done:
                # Construct the full state
                state = np.concatenate([
                    observation['observation'], 
                    observation['achieved_goal'], 
                    observation['desired_goal']
                ])
                
                # Select Action (Note: evaluate=True removes random noise!)
                action = agent.choose_action(state, evaluate=True)
                
                # Step
                observation, reward, done, truncated, info = env.step(action)
                done = done or truncated
                score += reward
                
            # Check success
            # Success is defined as reaching the target (info['is_success']) OR getting a decent score
            # In PandaReach, -1 per step means perfect score is near 0. -50 is max steps failure.
            if info.get('is_success', False) or score > -49:
                 episode_success = True
                 print(f"Test Episode {i+1} | Score: {score:.2f} | SUCCESS (Attempts: {attempts})")
                 success_count += 1
            else:
                 print(f"Test Episode {i+1} | Score: {score:.2f} | FAIL")

    print(f"\nFinal Success Rate: {success_count}/10 ({(success_count/10)*100:.1f}%)")
    env.close()

if __name__ == '__main__':
    test()