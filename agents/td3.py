import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from replay_memory.ReplayBuffer import ReplayBuffer
from utils.networks import ActorNetwork, CriticNetwork
from utils.HER import her_augmentation
from tensorflow.keras.optimizers import Adam

class TD3Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 update_actor_interval=2, n_actions=7, max_size=1000000, 
                 layer1_size=512, layer2_size=256, batch_size=256, noise=0.1, 
                 warmup=1000, model_name='TD3_HER_Panda'):
       
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.batch_size = batch_size
        self.noise = noise
        self.n_actions = n_actions
        self.warmup = warmup
        self.update_actor_iter = update_actor_interval
        self.learn_step_cntr = 0
        self.time_step = 0
        self.model_name = model_name
        self.env = env

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.her_augmentation = her_augmentation
        
        self.checkpoints_dir = 'ckp/'
        self.model_dir = os.path.join(self.checkpoints_dir, self.model_name)
        
        # 2. Create the directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.actor = ActorNetwork(n_actions=n_actions, name='actor', 
                                  model=model_name, checkpoints_dir='ckp/')
        
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor', 
                                         model=model_name, checkpoints_dir='ckp/')
        
        self.critic_1 = CriticNetwork(name='critic_1', model=model_name, 
                                      checkpoints_dir='ckp/')
        
        self.target_critic_1 = CriticNetwork(name='target_critic_1', model=model_name, 
                                             checkpoints_dir='ckp/')
        
        self.critic_2 = CriticNetwork(name='critic_2', model=model_name, 
                                      checkpoints_dir='ckp/')
        
        self.target_critic_2 = CriticNetwork(name='target_critic_2', model=model_name, 
                                             checkpoints_dir='ckp/')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        
        dummy_state = tf.ones([1, input_dims[0]], dtype=tf.float32)
        dummy_action = tf.ones([1, n_actions], dtype=tf.float32)
        
        # Build all 6 networks by calling them once with dummy data
        self.actor(dummy_state)
        self.target_actor(dummy_state)
        self.critic_1(dummy_state, dummy_action)
        self.target_critic_1(dummy_state, dummy_action)
        self.critic_2(dummy_state, dummy_action)
        self.target_critic_2(dummy_state, dummy_action)
        
        self.update_network_parameters(tau=1)
    


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation, evaluate=False):
        
        if not evaluate:
            if self.time_step < self.warmup:
                mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
                mu_prime = mu
            else:
                state = tf.convert_to_tensor([observation], dtype=tf.float32)
                mu = self.actor(state)[0] 
                mu_prime = mu + np.random.normal(scale=self.noise)
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0] 
            mu_prime = mu
    
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        self.time_step += 1

        return mu_prime.numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        
        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_1.set_weights(weights)
        
        
        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_2.set_weights(weights)

    @tf.function
    def learn(self):
        # 1. Start Training Check
        if self.memory.mem_cntr < self.batch_size * 2:
            return

        # 2. Sample Data
        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        # Convert NumPy arrays to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # --- CRITIC TRAINING ---
        with tf.GradientTape(persistent=True) as tape:
            # 3. Target Action Generation (TD3 Trick 3: Smoothing)
            # Get the action from the stable Target Actor
            target_actions = self.target_actor(new_states)
            
            # Add clipped noise to the target action for smoothing
            target_actions = target_actions + \
                tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
            
            # Ensure the noisy action stays within the environment's limits
            target_actions = tf.clip_by_value(target_actions, self.min_action, 
                                              self.max_action)

            # 4. Target Q-Value Calculation (TD3 Trick 1: Twin Critics)
            # Use the Target Critics to predict Q-values for the (S', A') transition
            q1_target_value = self.target_critic_1(new_states, target_actions)
            q2_target_value = self.target_critic_2(new_states, target_actions)
            
            # Use the minimum of the two targets to prevent overestimation
            q_target_min = tf.math.minimum(q1_target_value, q2_target_value)
            
            # Calculate the final Bellman target (R + gamma * min(Q))
            y = rewards + self.gamma * q_target_min * (1 - dones)
            
            # 5. Online Q-Value Prediction
            # Predict Q-values from the active Critics for the (S, A) transition
            q1_value = self.critic_1(states, actions)
            q2_value = self.critic_2(states, actions)
            
            # 6. Critic Loss (Mean Squared Error)
            critic_loss_1 = keras.losses.MSE(y, q1_value)
            critic_loss_2 = keras.losses.MSE(y, q2_value)

        # 7. Critic Optimization
        # Calculate gradients and apply them to the active Critic Networks
        critic_gradient_1 = tape.gradient(critic_loss_1, 
                                          self.critic_1.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_gradient_1, 
                                                    self.critic_1.trainable_variables))
        
        critic_gradient_2 = tape.gradient(critic_loss_2, 
                                          self.critic_2.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(critic_gradient_2, 
                                                    self.critic_2.trainable_variables))
        
        # --- ACTOR TRAINING (TD3 Trick 2: Delayed Update) ---
        self.learn_step_cntr += 1
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return # Skip Actor update unless the interval is met
            
        with tf.GradientTape() as tape:
            # 8. Actor Loss Calculation (Policy Gradient)
            # Get the action the Actor would choose for the current state
            new_actions = self.actor(states)
            
            # Evaluate that action using one of the Critics (e.g., Critic 1)
            actor_loss = -self.critic_1(states, new_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        # 9. Actor Optimization
        # Calculate gradients and apply them to the Actor Network
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, 
                                                 self.actor.trainable_variables))

        # 10. Target Network Update
        self.update_network_parameters()

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoints_file)
        self.target_actor.save_weights(self.target_actor.checkpoints_file)
        self.critic_1.save_weights(self.critic_1.checkpoints_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoints_file)
        self.critic_2.save_weights(self.critic_2.checkpoints_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoints_file)

    def load_models(self):
        print('... loading models ...')
        try:
            self.actor.load_weights(self.actor.checkpoints_file)
            self.target_actor.load_weights(self.target_actor.checkpoints_file)
            self.critic_1.load_weights(self.critic_1.checkpoints_file)
            self.target_critic_1.load_weights(self.target_critic_1.checkpoints_file)
            self.critic_2.load_weights(self.critic_2.checkpoints_file)
            self.target_critic_2.load_weights(self.target_critic_2.checkpoints_file)
            print("... models loaded successfully ...")
        except Exception as e:
            print(f"... no checkpoints found, starting from scratch ... Error: {e}")