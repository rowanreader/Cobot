# the actual RL part

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from MemoryBuffer import ReplayBuffer
from Networks import Actor, Critic
import numpy as np

# this is the parent class of Actor and Critic classes
class Agent:
    def __init__(self, inputShape, env, maxSize=1000000, dim1=512, dim2=512, numActions=3, batchSize=64, alpha=0.03,
                 beta=0.02, tau=0.005, gamma=0.99, noise=10):

        self.numActions = numActions
        self.memory = ReplayBuffer(maxSize, inputShape, numActions)
        self.noise = noise
        self.batchSize = batchSize # how many memories to sample
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(dim1=dim1, dim2=dim2, numActions=numActions)
        self.critic = Critic(dim1=dim1, dim2=dim2)
        self.targetActor = Actor(name="targetActor", dim1=dim1, dim2=dim2, numActions=numActions)
        self.targetCritic = Critic(name="targetCritic", dim1=dim1, dim2=dim2)

        self.maxAction = env.action_space.high[0]
        self.minAction = env.action_space.low[0]

        # configure the model with losses and metrics
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.targetActor.compile(optimizer=Adam(learning_rate=alpha))
        self.targetCritic.compile(optimizer=Adam(learning_rate=beta))


        self.updateNets(self.tau) # ???

    def updateNets(self, tau):

        weights = []
        targets = self.targetActor.weights # get current weights of target actor
        for i, weight in enumerate(self.actor.weights): # for each weight in actor (i is 0-num)
            weights.append(weight*tau + targets[i]*(1-tau)) # update current weight based on target weights
        self.targetActor.set_weights(weights) # update target actor

        # do the same for the critic network
        weights = []
        targets = self.targetCritic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.targetCritic.set_weights(weights)

    # to store the specific state transition
    def record(self, state, action, reward, state_, endFlag, obsLen):

        state = np.reshape(state, [obsLen], order='C')

        state_ = np.reshape(state_, [obsLen], order='C')
        self.memory.store(state, action, reward, state_, endFlag)

    def saveModels(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.saveFile)
        self.targetActor.save_weights(self.targetActor.saveFile)
        self.critic.save_weights(self.critic.saveFile)
        self.targetCritic.save_weights(self.targetCritic.saveFile)

    def loadModels(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.saveFile)
        self.targetActor.load_weights(self.targetActor.saveFile)
        self.critic.load_weights(self.critic.saveFile)
        self.targetCritic.load_weights(self.targetCritic.saveFile)

    # based on current state, choose action
    # if train is true, add noise to simulate reality, if false, don't
    def chooseAction(self, observation, obsLen, evaluate=True):
        observation = np.reshape(observation, [obsLen], order='C')
        state = tf.convert_to_tensor([observation], dtype=tf.float32) # the form needed for submitting to net

        # get actions = degrees to rotate for all joints
        actions = self.actor(state) # automatically calls 'call' function
        if not evaluate:
            actions += tf.random.normal(shape=[self.numActions], mean=0, stddev=self.noise)

        actions = tf.clip_by_value(actions, [self.minAction,self.minAction,self.minAction], [self.maxAction,self.maxAction,self.maxAction])


        if np.isnan(actions[0][0]):
            print("Oops")
        return actions[0].numpy()

    def learn(self):

        # if memory bank not sufficiently filled, just return
        if self.memory.counter < self.batchSize:
            return
        state, action, reward, state_, flag = self.memory.sample(self.batchSize)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        states_ = tf.convert_to_tensor(state_, dtype=tf.float32)

        with tf.GradientTape() as tape:
            targetActions = self.targetActor(states_) # get actions chosen by target Actor when presented new state
            # based on targetActions, find targetCritic's evaluation
            criticVal_ = tf.squeeze(self.targetCritic(states_, targetActions), 1) # squeeze along 1st dim

            # do same for old state based on actions already chosen
            criticVal = tf.squeeze(self.critic(states, actions), 1)

            # update target value as reward with some influence from critic
            target = reward + self.gamma*criticVal_*(1-flag) # only update if not a terminal episode

            criticLoss = keras.losses.MSE(target, criticVal) # get loss based on target value and what was predicted

        critNetGradient = tape.gradient(criticLoss, self.critic.trainable_variables) # get loss gradient, minimize?
        self.critic.optimizer.apply_gradients(zip(critNetGradient, self.critic.trainable_variables))

        # do same for action
        with tf.GradientTape() as tape:
            actorActions = self.actor(states) # get actions
            actorLoss = -self.critic(states, actorActions) # get critic's q values based on states and new actions
            actorLoss = tf.math.reduce_mean(actorLoss) # calculate mean?

        actNetGradient = tape.gradient(actorLoss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actNetGradient, self.actor.trainable_variables))

        self.updateNets(self.tau)







