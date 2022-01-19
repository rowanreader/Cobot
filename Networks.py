# Has Actor agent and Critic agent classes

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten


# critic network, child class of the keras Model class
# going to have 2 dense layers then output layer
class Critic(keras.Model):

    def __init__(self, name='critic', dim1=512, dim2=512):
        # call parent constructor
        super(Critic, self).__init__() # not entirely clear why we call parent constructor
        self.dim1 = dim1
        self.dim2 = dim2
        self.modelName = name  # probably not needed
        self.saveFile = self.modelName + ".h5"

        self.layer1 = Dense(self.dim1, activation='relu')
        self.layer2 = Dense(self.dim2, activation='relu')
        self.q = Dense(1, activation=None) # output is q-value


    # apply layers to input
    # takes in state and action to be used, generates Q value
    # must do it this way, because we don't compile model or train on set of data
    def call(self, state, action):
        # combine the state and action together via concatenation
        # print(state)
        # print(action)
        qVal = self.layer1(tf.concat([state, action], axis=1))
        qVal = self.layer2(qVal)
        qVal = self.q(qVal)

        return qVal


# make actor network
# similar to critic
class Actor(keras.Model):

    # takes in dimensions for 2 layers and number of actions (x,y,x -> needed for output layer)
    def __init__(self, name='actor', dim1=512, dim2=512, numActions=3):
        super(Actor, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.numActions = numActions
        self.modelName = name
        self.saveFile = self.modelName + ".h5"

        # self.flatten = Flatten()
        self.layer1 = Dense(self.dim1, activation='relu')
        self.layer2 = Dense(self.dim2, activation='relu')
        self.out = Dense(self.numActions) # default activation is linear/identity

    def call(self, state): # should this be __call__?
        # state = self.flatten(state)
        action = self.layer1(state)
        action = self.layer2(action)
        action = self.out(action)
        return action



