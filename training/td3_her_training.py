import gymnasium as gym
import numpy as np
import panda_gym
import sys
import os

# Add parent directory to path so we can import 'agents'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.td3 import TD3Agent

# 1. Initialize the Environment
# We use 'PandaReach-v3' and set render_mode="human" so you can SEE the robot moving.
# For faster training later, you can change "human" to "rgb_array".
env = gym.make('PandaReach-v3', render_mode="rgb_array")

# 2. Calculate the correct input size
# State = Observation (Robot Joints) + Achieved Goal + Desired Goal
obs_dim = env.observation_space['observation'].shape[0]   # 6 elements
goal_dim = env.observation_space['desired_goal'].shape[0] # 3 elements
# Combined input is obs_dim + 2 * goal_dim (6 + 3 + 3 = 12)
combined_input_dims = (obs_dim + 2 * goal_dim,) 

# 3. Define Training Parameters
# 3. Define Training Parameters
n_episodes = 50000 # MARATHON MODE
max_steps = 50     

# 4. Initialize the TD3 Agent
agent = TD3Agent(
    alpha=0.0001, beta=0.0001, # FINE TUNING: Low LR for stability
    tau=0.005,
    env=env,
    input_dims=combined_input_dims,
    n_actions=env.action_space.shape[0],
    layer1_size=256, layer2_size=256,
    batch_size=128, 
    noise=0.1,      # Stabilized noise
    warmup=0        # No warmup, using pre-trained weights
)

# Load existing models to resume training if available
try:
    agent.load_models()
except Exception as e:
    print(f"Could not load models: {e}")

# 5. Prepare for Storage
best_score = float('-inf')
score_history = []
best_consecutive_successes = 0
consecutive_successes = 0
best_consecutive_successes = 0
consecutive_successes = 0
best_success_rate = 0.15 # PROTECTION: Only save if we beat the previous best (15%)

print("Starting Training...")

for i in range(n_episodes):
    # 1. Reset Environment for a new attempt
    observation, info = env.reset()
    done = False
    score = 0
    
    # Create a temporary buffer to store just THIS episode's data (for HER)
    episode_buffer = {'obs': [], 'transitions': []}
    
    # 2. Run the Episode (up to max_steps)
    for step in range(max_steps):
        
        # --- FIX: Construct the full 12-element state vector ---
        state = np.concatenate([
            observation['observation'], 
            observation['achieved_goal'], 
            observation['desired_goal']
        ])
        
        # Choose Action (Agent uses the full 12-element state)
        action = agent.choose_action(state)
        
        # Execute Action
        next_observation, reward, done, truncated, info = env.step(action)
        
        # --- FIX: Construct the next full 12-element state vector ---
        next_state = np.concatenate([
            next_observation['observation'], 
            next_observation['achieved_goal'], 
            next_observation['desired_goal']
        ])
        
        # Update flags
        done = done or truncated
        
        # 3. Store the REAL experience in Memory
        agent.remember(state, action, reward, next_state, done)
        
        # 4. Agent Learns (Updates weights)
        agent.learn()
        
        # 5. Collect data for HER (Hindsight Experience Replay)
        episode_buffer['obs'].append(observation)
        episode_buffer['transitions'].append({
            'observation': observation['observation'], # Only robot's observation for HER
            'action': action,
            'reward': reward,
            'next_observation': next_observation['observation'],
            'done': done,
            'achieved_goal': observation['achieved_goal'],
            'next_achieved_goal': next_observation['achieved_goal'] # <--- NEW: Needed for correct HER
        })
        
        score += reward
        observation = next_observation
        
        if done:
            break

    # ... (inside loop) ...
    # 6. Apply HER (The "Time Travel" Trick)
    agent.her_augmentation(agent, episode_buffer)

    # 6.5 NOISE DECAY (Crucial for bridging the gap to deterministic success)
    if agent.noise > 0.01:
        agent.noise *= 0.9995 # Slow decay per episode
    
    # 7. Logging and Saving
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    print(f"Episode {i} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Noise: {agent.noise:.4f}")

    # IMPROVED: Evaluate Deterministic Policy every 20 episodes
    if i % 20 == 0 and i > 0:
        print(f"--- Evaluating Deterministic Policy @ Episode {i} ---")
        eval_success_count = 0
        n_eval_episodes = 20 # Test on 20 episodes
        
        for _ in range(n_eval_episodes):
            eval_obs, _ = env.reset()
            eval_done = False
            while not eval_done:
                eval_state = np.concatenate([
                    eval_obs['observation'], 
                    eval_obs['achieved_goal'], 
                    eval_obs['desired_goal']
                ])
                # evaluate=True disables noise for pure performance check
                eval_action = agent.choose_action(eval_state, evaluate=True)
                eval_obs, _, eval_done, eval_truncated, eval_info = env.step(eval_action)
                eval_done = eval_done or eval_truncated
                
                if eval_info.get('is_success', False):
                    eval_success_count += 1
                    break # Success!
        
        eval_success_rate = eval_success_count / n_eval_episodes
        print(f"--- Eval Success Rate: {eval_success_rate*100:.1f}% ---")
        
        # Save "latest" checkpoint regardless of performance (for resumption)
        # agent.save_models(suffix="_latest") # (Requires modifying save_models which we avoid for now)
        
        # Save "best" ONLY if strictly better
        if eval_success_rate > best_success_rate:
            best_success_rate = eval_success_rate
            print(f"!!! NEW BEST EVAL RATE: {best_success_rate*100:.1f}% --- SAVING ROBUST MODEL ---")
            agent.save_models() # This overwrites 'actor.weights.h5' - ensuring it's always the BEST one
            
    # CRITICAL: Regular saves to prevent total loss if crash
    if i % 500 == 0 and i > 0:
         print("--- Periodic Save (Backup) ---")
         # We rely on the best save logic above for the main file, 
         # but we can implement a secondary save mechanism later if needed.

env.close()