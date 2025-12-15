import numpy as np

def her_augmentation(agent, episode_buffer):
    k = 4
    
    trajectory = episode_buffer['obs']
    
    compute_reward = agent.env.unwrapped.compute_reward

    for index, transition in enumerate(episode_buffer['transitions']):
        
        for _ in range(k):
            
            future_index = np.random.randint(index, len(trajectory))
            new_goal = trajectory[future_index]['achieved_goal']
            
            new_state = np.concatenate([
                transition['observation'], 
                transition['achieved_goal'], 
                new_goal
            ])

            # --- FIX: Reconstruct the next state using the NEW GOAL ---
            # Next State = Next Observation + Next Achieved Goal + NEW Desired Goal
            new_next_state = np.concatenate([
                transition['next_observation'],
                transition['next_achieved_goal'],
                new_goal
            ])
            reward = compute_reward(transition['achieved_goal'], new_goal, {})

            # Recompute 'done' based on the new reward
            # In PandaReach, 0 reward means success (done)
            done = True if reward == 0 else False
            
            agent.remember(
                new_state, 
                transition['action'], 
                reward, 
                new_next_state, 
                done
            )