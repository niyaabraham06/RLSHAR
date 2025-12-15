import gymnasium as gym
import numpy as np
import panda_gym
from agents.td3 import TD3Agent

def test():
    # --- CRITICAL FIX: Use reward_type="dense" to match your training! ---
    env = gym.make('PandaReach-v3', render_mode="human", reward_type="dense")

    # Dimensions
    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    combined_input_dims = (obs_dim + 2 * goal_dim,)

    # Initialize Agent
    agent = TD3Agent(
        alpha=0.001, beta=0.001, tau=0.005,
        env=env, input_dims=combined_input_dims,
        n_actions=env.action_space.shape[0],
        layer1_size=256, layer2_size=256,
        batch_size=256
    )

    # Load the best brain
    agent.load_models()

    print("\n--- Starting Test (Dense Rewards Mode) ---")
    
    success_count = 0
    total_episodes = 10

    for i in range(total_episodes):
        observation, info = env.reset()
        done = False
        score = 0
        
        while not done:
            state = np.concatenate([
                observation['observation'], 
                observation['achieved_goal'], 
                observation['desired_goal']
            ])
            
            # Select Action (Deterministic = No random noise)
            action = agent.choose_action(state, evaluate=True)
            
            observation, reward, done, truncated, info = env.step(action)
            done = done or truncated
            score += reward
            
            # Optional: Add a tiny delay if you want to watch it in slow-mo
            # import time; time.sleep(0.01)

        # In Dense mode, a score better than -10 usually means it reached the target
        # -3.0 or -4.0 is basically perfect.
        is_success = score > -10
        if is_success:
            success_count += 1
            
        status = "SUCCESS" if is_success else "FAIL"
        print(f"Test Episode {i+1} | Score: {score:.2f} | {status}")

    print(f"\nFinal Success Rate: {success_count}/{total_episodes} ({(success_count/total_episodes)*100:.1f}%)")
    env.close()

if __name__ == '__main__':
    test()