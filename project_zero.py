import gymnasium as gym
import numpy as np
import panda_gym
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import os

# ==========================================
# 🧠 PART 1: THE BRAIN (TD3 AGENT & BUFFER)
# ==========================================

class ReplayBuffer:
    # (ReplayBuffer class code unchanged)
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
    # (TD3Agent class code unchanged)
    def __init__(self, input_dims, n_actions, env):
        self.env = env
        self.n_actions = n_actions
        self.batch_size = 256
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        # Build Networks
        self.actor = self.build_actor(input_dims, n_actions)
        self.target_actor = self.build_actor(input_dims, n_actions)
        self.critic_1 = self.build_critic(input_dims, n_actions)
        self.target_critic_1 = self.build_critic(input_dims, n_actions)
        self.critic_2 = self.build_critic(input_dims, n_actions)
        self.target_critic_2 = self.build_critic(input_dims, n_actions)

        self.actor.compile(optimizer=Adam(learning_rate=0.001))
        self.critic_1.compile(optimizer=Adam(learning_rate=0.001))
        self.critic_2.compile(optimizer=Adam(learning_rate=0.001))
        
        # Initialize Targets
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.memory = ReplayBuffer(100000, input_dims, n_actions)

    def build_actor(self, input_dims, n_actions):
        inputs = keras.Input(shape=input_dims)
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_actions, activation='tanh')(x)
        outputs = x * self.env.action_space.high[0]
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
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)
        return np.clip(actions.numpy()[0], self.min_action, self.max_action)

    @tf.function
    def learn(self):
        if self.memory.mem_cntr < self.batch_size: return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(new_states)
            target_actions += tf.clip_by_value(tf.random.normal(shape=[self.batch_size, self.n_actions], stddev=0.2), -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            q1_ = self.target_critic_1([new_states, target_actions])
            q2_ = self.target_critic_2([new_states, target_actions])
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)
            
            target = rewards + 0.99 * tf.minimum(q1_, q2_) * (1 - dones.astype(float))
            
            q1 = tf.squeeze(self.critic_1([states, actions]), 1)
            q2 = tf.squeeze(self.critic_2([states, actions]), 1)
            
            critic_loss_1 = keras.losses.MSE(target, q1)
            critic_loss_2 = keras.losses.MSE(target, q2)

        grads_1 = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))
        grads_2 = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = -tf.math.reduce_mean(self.critic_1([states, new_actions]))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_targets(self.target_actor.variables, self.actor.variables)
        self.update_targets(self.target_critic_1.variables, self.critic_1.variables)
        self.update_targets(self.target_critic_2.variables, self.critic_2.variables)

    def update_targets(self, target_weights, weights, tau=0.005):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

# ==========================================
# 🚀 PART 2: THE EXECUTION
# ==========================================

def run_project_zero():
    # --- PHASE 1: TRAINING (Forced 50 Episodes) ---
    print("\n" + "="*40)
    print("🚀 TRAINING STARTED (FORCED 50 EPISODES)")
    print("This will guarantee a diverse trained brain.")
    print("="*40)
    
    env = gym.make('PandaReach-v3', render_mode="rgb_array", reward_type="dense")
    
    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    input_dims = (obs_dim + 2 * goal_dim,)
    n_actions = env.action_space.shape[0]

    agent = TD3Agent(input_dims, n_actions, env)
    
    # FORCED TRAINING COUNT (50 episodes minimum practice)
    MIN_TRAINING_EPISODES = 50 
    
    for i in range(200): # Max 200 episodes
        obs, _ = env.reset()
        done = False
        score = 0
        
        for _ in range(50):
            state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
            action = agent.choose_action(state)
            obs, reward, done, truncated, _ = env.step(action)
            
            next_state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
            done = done or truncated
            
            agent.memory.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            score += reward
            if done: break
            
        print(f"Training Ep {i+1} | Score: {score:.1f}")
        
        # ONLY START DEMO AFTER MINIMUM PRACTICE IS COMPLETE
        # AND the score is decent (Score > -5.0)
        if i >= MIN_TRAINING_EPISODES - 1 and score > -5.0: 
            print(f"\n✅ BRAIN TRAINED! (Score {score:.2f})")
            print("Swapping to DEMO MODE now...")
            break
        elif i >= 199:
             print("\nTraining complete (Max episodes reached). Starting Demo.")

    env.close()

    # --- PHASE 2: LIVE DEMO (GUI) ---
    print("\n" + "="*40)
    print("🎥 LIVE DEMO STARTING")
    print("="*40)
    
    env_demo = gym.make('PandaReach-v3', render_mode="rgb_array", reward_type="dense")
    
    while True:
        obs, _ = env_demo.reset()
        done = False
        
        print("\nTarget Appeared. Moving...")
        
        for step in range(80):
            state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
            
            action = agent.choose_action(state, evaluate=True)
            obs, _, done, truncated, _ = env_demo.step(action)
            
            if done or truncated: break
            
            time.sleep(0.01) # Small sleep for natural visual speed
            
        # Check Success
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) * 100
        if dist < 5.0:
            print(f"✅ HIT! Distance: {dist:.1f} cm")
        else:
            print(f"❌ Miss. Distance: {dist:.1f} cm")
            
        time.sleep(1.0) # Pause between targets

if __name__ == '__main__':
    run_project_zero()