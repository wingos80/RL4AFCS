import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import time
import csv
import sys
from tiles import *
import matplotlib.pyplot as plt

class SARSA:
    # Episodic Semi-gradient Sarsa with linear function approximator for action-value function
    def __init__(self, env, alpha=0.5, gamma=1, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = env.action_space.n

        self.d = 8 # number of tilings and dimension of function approximation
        self.weights = np.zeros((self.d,1))
        self.iht = IHT(4096)

        self._set_up_plot()

    def _feat_vec(self, state, action):
        """
        Feature vector

        :param state: current state
        :param action: action to take
        :return: feature vector
        """
        x     = state[0]
        x_dot = state[1]
        # print("x",x)
        # print("x_dot",x_dot)
        feature_vector = tiles(self.iht, self.d, [self.d*x/(0.5+1.2), self.d*x_dot/(0.07*2)],[action])
        return feature_vector.reshape((-1,1)) # making sure the feature vector is a column vector
    
    def _grad_Q(self, state, action):
        """
        Gradient of action-value function

        :param state: current state
        :param action: action to take
        :return: gradient of action-value function
        """
        return self._feat_vec(state, action)
    
    def _Q(self, state, action):
        """
        Action-value function

        :param state: current state
        :param action: action to take
        :return: Q value of state-action pair
        """
        return np.dot(self.weights.T, self._feat_vec(state, action))
    
    def _policy(self, state):
        """
        Policy, returns the probablitiy of taking each action

        :param state: current state
        :return: array of probabilities
        """
        N = self.N

        Q = []
        for action in range(N):
            Q.append(self._Q(state, action))
        Q = np.array(Q)
        best_actions = np.flatnonzero(Q==Q.max())
        num_equal_contenders = np.size(best_actions)
        num_loser = N - num_equal_contenders

        p = np.zeros(N)
        for i in range(N):
            if i in best_actions:
                p[i] = (1 - self.epsilon)/num_equal_contenders
            else:
                p[i] = self.epsilon/num_loser

        return p/sum(p)
    
    def _sample_action(self, state):
        """
        Sample an action from the policy

        :param state: current state
        :return: action to take
        """
        actions = np.arange(self.N)
        return np.random.choice(actions, p=self._policy(state))
    
    def _set_up_plot(self):
        self.fig, self.ax = plt.subplots()
        # set the xlim to between 0 and the max number of episodes
        # self.ax.set_xlim(-10, self.num_episodes+10)
        self.ax.set_xlabel("Time steps")
        self.ax.set_ylabel("Episodes")
        # self.ax.set_title("Currently at Episode 0 (reward=?)")
        self.fig.show()

    def _update_plot(self):    
        self.ax.clear()
        self.ax.plot(self.timesteps, self.episodes)
        self.ax.set_xlabel("Time steps")
        self.ax.set_ylabel("Episodes") 
        # self.ax.set_title(f"Currently at Episode {episode} (reward={self.rewards_list[-1]})")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def train(self, max_episodes=100):

        timestep  = 0
        self.timesteps = [timestep]
        self.episodes  = [1]
        self._update_plot()
        for episode in range(max_episodes):
            done = False
            np.random.seed(episode)
            obs = self.env.reset(seed=episode)
            states = obs[0]
            action = self._sample_action(states)
            while not done:
                env.render()
                next_obs, reward, done, _, _ = self.env.step(action)
                next_states = next_obs.copy()
                if done:
                    self.weights += self.alpha * (reward - self._Q(states,action))*self._grad_Q(states, action)
                next_action = self._sample_action(next_states)
                td_error = reward + self.gamma * self._Q(next_states, next_action) - self._Q(states, action)
                self.weights += self.alpha * td_error * self._grad_Q(states, action)
                states = next_states.copy()
                action = next_action

                timestep += 1
                if timestep % 20 == 0:
                    print("actino: ", action)
                    print("position: ", states[0], "\n\n")
                    print("time step: ", timestep)

            if done:
                self.timesteps.append(timestep)
                self.episodes.append(episode+1)
                self.timesteps.append(timestep)
                self.episodes.append(episode+2)
                self._update_plot()



env = gym.make("MountainCar-v0",render_mode="human")
obs = env.reset()

sarsa = SARSA(env, alpha=0.5, gamma=1, epsilon=0.1)
sarsa.train(max_episodes=100)

# for i in range(100):
#     action = env.action_space.sample()
#     obs, reward, done, _, _ = env.step(action)
#     print("actino:",action)
#     print("opps",obs)
#     env.render()
#     # print('hek')