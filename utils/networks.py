import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Concatenate

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, name, model, checkpoints_dir="ckp/"):
        super(ActorNetwork, self).__init__()
        
        self.checkpoints_file = os.path.join(checkpoints_dir, model, name + ".weights.h5") 

        self.layer1 = Dense(512, activation="relu")
        self.layer2 = Dense(256, activation="relu")
        self.layer3 = Dense(256, activation="relu")

        self.action_output = Dense(n_actions, activation="tanh")

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)

        actions = self.action_output(x)
        return actions


class CriticNetwork(keras.Model):
    def __init__(self, name, model, checkpoints_dir="ckp/"):
        super(CriticNetwork, self).__init__()

        self.checkpoints_file = os.path.join(checkpoints_dir, model, name + ".weights.h5")

        self.concat = Concatenate(axis=1)

        self.layer1 = Dense(512, activation="relu")
        self.layer2 = Dense(256, activation="relu")
        self.layer3 = Dense(256, activation="relu")

        self.q_output = Dense(1, activation=None)

    def call(self, state, action):

        x = self.concat([state, action])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        q_value = self.q_output(x)
        return q_value