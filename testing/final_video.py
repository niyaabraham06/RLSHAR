import gymnasium as gym
import numpy as np
import panda_gym
from agents.td3 import TD3Agent
import time

def final_video():
    # 1. Use Dense Rewards (This matches your trained model)
    env = gym.make('PandaReach-v3', render_mode="human", reward_type="dense")

    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    combined_input_dims = (obs_dim + 2 * goal_dim,)

    # 2. Load the Agent
    agent = TD3Agent(
        alpha=0.001, beta=0.001, tau=0.005,
        env=env, input_dims=combined_input_dims,
        n_actions=env.action_space.shape[0],
        layer1_size=256, layer2_size=256,
        batch_size=256
    )
    agent.load_models()

    print("\n--- FINAL VIDEO MODE ---")
    print("Running Raw AI in Slow Motion (No Math Smoothing)")
    print("Press Ctrl+C to stop.\n")

    episode = 1
    while True:
        observation, info = env.reset()
        done = False
        score = 0
        
        print(f"Episode {episode} running...")
        
        while not done:
            state = np.concatenate([
                observation['observation'], 
                observation['achieved_goal'], 
                observation['desired_goal']
            ])
            
            # Use the Raw AI (evaluate=True means no random noise)
            action = agent.choose_action(state, evaluate=True)
            
            # Execute
            observation, reward, done, truncated, info = env.step(action)
            
            # --- THE TRICK: SLOW MOTION ---
            # We pause for 40ms. This removes the "jitter" perception 
            # without breaking the robot's control loop.
            time.sleep(0.04) 
            
            score += reward
            done = done or truncated

        # Check result
        dist = np.linalg.norm(observation['achieved_goal'] - observation['desired_goal'])
        dist_cm = dist * 100
        
        if dist_cm < 5.0:
            print(f"✅ SUCCESS! Distance: {dist_cm:.1f} cm (Score: {score:.1f})")
            print(">>> GOOD TAKE! SAVE THIS VIDEO <<<")
            time.sleep(2) # Pause to let you stop recording
        else:
            print(f"❌ Miss. Distance: {dist_cm:.1f} cm")
            time.sleep(0.5)
            
        episode += 1

    env.close()

if __name__ == '__main__':
    final_video()