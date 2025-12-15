import numpy as np

def her_augmentation(agent, episode_buffer, k=4):
    """
    Hindsight Experience Replay (HER)
    Creates 'k' virtual successful transitions for every real transition.
    """
    obs_list = episode_buffer['obs']
    transitions = episode_buffer['transitions']
    
    # Iterate through the episode to create virtual experiences
    for t, transition in enumerate(transitions):
        
        # We want to generate 'k' future goals for every step
        future_indices = np.random.randint(t, len(transitions), size=k)
        
        for future_t in future_indices:
            # 1. The Trick: Pretend the goal we wanted to reach was actually
            #    the state we achieved at some future point (future_t).
            future_transition = transitions[future_t]
            new_goal = future_transition['achieved_goal']
            
            # 2. Reconstruct the State with the NEW Goal
            # Original State: [Robot_Joints, Achieved_Goal, OLD_Goal]
            # New State:      [Robot_Joints, Achieved_Goal, NEW_Goal]
            
            # Current State Relabeled
            current_obs = obs_list[t]
            new_state = np.concatenate([
                current_obs['observation'],
                current_obs['achieved_goal'],
                new_goal  # <--- The Magic Swap
            ])
            
            # Next State Relabeled
            next_obs_dict = obs_list[t+1] # obs_list has t+1 elements
            new_next_state = np.concatenate([
                next_obs_dict['observation'],
                next_obs_dict['achieved_goal'],
                new_goal  # <--- The Magic Swap
            ])
            
            # 3. CRITICAL: Re-Calculate Reward
            # We must ask the environment: "If this was my goal, what would the reward be?"
            # Since we are using Sparse Rewards, this should return 0.0 (Success) or -1.0 (Fail)
            
            # --- FIX: Use .unwrapped to access the hidden compute_reward function ---
            info = {} 
            new_reward = agent.env.unwrapped.compute_reward(
                achieved_goal=np.array([transition['achieved_goal']]), 
                desired_goal=np.array([new_goal]), 
                info=info
            )
            # Extract float from the result (compute_reward returns an array)
            new_reward = float(new_reward)

            # 4. Store this "Fake" Success in Memory
            # We treat it exactly like a real experience
            agent.remember(
                new_state, 
                transition['action'], 
                new_reward, 
                new_next_state, 
                True # Done (conceptually, hitting the goal ends the task)
            )