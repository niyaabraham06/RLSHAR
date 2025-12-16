import gymnasium as gym
import numpy as np
import panda_gym
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time

# ==========================================
#  PART 1: THE BRAIN (TD3 AGENT & BUFFER)
# ==========================================

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (self.state_memory[batch], self.action_memory[batch], 
                self.reward_memory[batch], self.new_state_memory[batch], 
                self.terminal_memory[batch])

class TD3Agent:
    def __init__(self, input_dims, n_actions, env):
        self.env = env
        self.n_actions = n_actions
        self.batch_size = 256
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.learn_step_cntr = 0 
        self.update_period = 2    
        
        # Build Networks
        self.actor = self.build_actor(input_dims, n_actions)
        self.target_actor = self.build_actor(input_dims, n_actions)
        self.critic_1 = self.build_critic(input_dims, n_actions)
        self.target_critic_1 = self.build_critic(input_dims, n_actions)
        self.critic_2 = self.build_critic(input_dims, n_actions)
        self.target_critic_2 = self.build_critic(input_dims, n_actions)

        # Learning rate tuned for stability
        lr = 0.0003 
        self.actor.compile(optimizer=Adam(learning_rate=lr))
        self.critic_1.compile(optimizer=Adam(learning_rate=lr))
        self.critic_2.compile(optimizer=Adam(learning_rate=lr))
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.memory = ReplayBuffer(100000, input_dims, n_actions)

    def build_actor(self, input_dims, n_actions):
        inputs = keras.Input(shape=input_dims)
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_actions, activation='tanh')(x)
        outputs = x * self.max_action
        return keras.Model(inputs, outputs)

    def build_critic(self, input_dims, n_actions):
        state_input = keras.Input(shape=input_dims)
        action_input = keras.Input(shape=(n_actions,))
        x = keras.layers.Concatenate()([state_input, action_input])
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        outputs = Dense(1)(x)
        return keras.Model([state_input, action_input], outputs)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            # Add exploration noise
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)
        return np.clip(actions.numpy()[0], self.min_action, self.max_action)

    # REMOVED @tf.function to prevent the Adam Optimizer variable creation error
    def learn(self):
        if self.memory.mem_cntr < self.batch_size: return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        # --- CRITIC UPDATE ---
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(new_states)
            # Target policy smoothing
            noise = tf.clip_by_value(tf.random.normal(shape=[self.batch_size, self.n_actions], stddev=0.2), -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions + noise, self.min_action, self.max_action)

            q1_ = self.target_critic_1([new_states, target_actions])
            q2_ = self.target_critic_2([new_states, target_actions])
            
            # Squeeze and calculate target Q-value (Bellman equation)
            target = rewards + 0.99 * tf.minimum(tf.squeeze(q1_), tf.squeeze(q2_)) * (1 - tf.cast(dones, tf.float32))
            
            q1 = tf.squeeze(self.critic_1([states, actions]))
            q2 = tf.squeeze(self.critic_2([states, actions]))
            
            critic_loss_1 = keras.losses.MSE(target, q1)
            critic_loss_2 = keras.losses.MSE(target, q2)

        grads_1 = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))
        grads_2 = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))

        self.learn_step_cntr += 1

        # --- DELAYED ACTOR UPDATE ---
        if self.learn_step_cntr % self.update_period == 0:
            with tf.GradientTape() as tape:
                new_actions = self.actor(states)
                actor_loss = -tf.math.reduce_mean(self.critic_1([states, new_actions]))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # Update target networks
            self.update_targets(self.target_actor.variables, self.actor.variables)
            self.update_targets(self.target_critic_1.variables, self.critic_1.variables)
            self.update_targets(self.target_critic_2.variables, self.critic_2.variables)

    def update_targets(self, target_weights, weights, tau=0.005):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

# ==========================================
#  PART 2: THE EXECUTION
# ==========================================

def run_project_zero():
    env = gym.make('PandaReach-v3', render_mode="rgb_array", reward_type="dense")
    
    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    input_dims = (obs_dim + 2 * goal_dim,)
    n_actions = env.action_space.shape[0]

    agent = TD3Agent(input_dims, n_actions, env)
    
    TOTAL_STEPS = 0
    WARMUP_STEPS = 10000 
    MAX_EPISODES = 600  
    
    print(f"Starting Warmup: Collecting {WARMUP_STEPS} random steps...")

    for i in range(MAX_EPISODES):
        obs, _ = env.reset()
        score = 0
        
        # 100 steps gives the arm ample time to correct its trajectory
        for _ in range(100):
            state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
            
            if TOTAL_STEPS < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)
                
            obs, reward, done, truncated, _ = env.step(action)
            next_state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
            
            agent.memory.store_transition(state, action, reward, next_state, done or truncated)
            
            if TOTAL_STEPS >= WARMUP_STEPS:
                agent.learn()
            
            score += reward
            TOTAL_STEPS += 1
            if done or truncated: break
            
        if (i + 1) % 10 == 0:
            status = "WARMUP" if TOTAL_STEPS < WARMUP_STEPS else "TRAINING"
            print(f"Episode {i+1} | {status} | Score: {score:.2f} | Total Steps: {TOTAL_STEPS}")

        # REFINED EXIT CONDITION:
        # Requires warmup finished AND at least 350 post-warmup episodes (approx i > 500 total)
        if TOTAL_STEPS > WARMUP_STEPS and i > 500 and score > -0.05:
            print(f"\nGoal Reached and Brain Trained! Final Score: {score:.2f}")
            break

    env.close()

    # --- LIVE DEMO ---
    print("\nStarting Demo (Press Ctrl+C to stop)...")
    env_demo = gym.make('PandaReach-v3', render_mode="human", reward_type="dense")
    
    try:
        while True:
            obs, _ = env_demo.reset()
            for _ in range(100):
                state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
                action = agent.choose_action(state, evaluate=True)
                obs, _, done, truncated, _ = env_demo.step(action)
                if done or truncated: break
                time.sleep(0.01)
            
            dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) * 100
            print(f"Result: {'HIT' if dist < 5.0 else 'Miss'} ({dist:.1f} cm)")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nDemo stopped.")
        env_demo.close()

if __name__ == '__main__':
    run_project_zero()