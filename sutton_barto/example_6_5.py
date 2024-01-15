import numpy as np
import pygame
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from GridWorldv1 import GridWorldEnv

np.random.seed(42)

gym.register(
     id="GridWorld-v0",
     entry_point="GridWorldv1:GridWorldEnv",
     max_episode_steps=300,
)

env = gym.make("GridWorld-v0",render_mode="human")
obs = env.reset()

class SARSA:
    def __init__(self, env, alpha=0.5, gamma=1, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = env.action_space.n
        high = env.observation_space['agent'].high
        low = env.observation_space['agent'].low
        steps = high - low + 1
        self.Q = np.zeros((steps[0], steps[1], self.N))

        self._set_up_plot()

    def _policy(self, state):
        """
        Policy, returns the probablitiy of taking each action

        :param state: current state
        :return: array of probabilities
        """
        N = self.N

        Q = self.Q[state[0], state[1]]
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
    
    
    def _Q_eps_geedy(self, epsilon, state):
        """
        Epsilon greedy policy, picks the action with the highest Q value with 1- epsilon probability
        and a random action with epsilon probability

        :param state: current state
        :return: action to take
        """
        Q = self.Q[state[0], state[1]]
        if np.random.random() < epsilon:
            return np.random.randint(0, self.N)
        else:
            return np.random.choice(np.flatnonzero(Q == Q.max()))

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
        """
        Train the agent for the given number of episodes

        :param episodes: number of episodes to train
        :return: None
        """
        timestep  = 0
        self.timesteps = [timestep]
        self.episodes  = [1]
        for episode in range(max_episodes):
            done = False
            obs, _ = self.env.reset()
            state = obs['agent'].copy()
            action = self._Q_eps_geedy(0, state)
            while not done:
                self.env.render()
                timestep += 1
                obs, reward, done, _, _ = self.env.step(action)
                next_state = obs['agent'].copy()
                # next_action = self._Q_eps_geedy(self.epsilon, next_state)
                next_action = self._sample_action(next_state)
                expected_Q = np.sum(self._policy(next_state) * self.Q[next_state[0], next_state[1]])
                SARSA_Q = self.Q[next_state[0], next_state[1], next_action]
                learning_Q = self.Q[next_state[0], next_state[1], self._Q_eps_geedy(0, next_state)]
                self.Q[state[0], state[1], action] += self.alpha * (
                            reward + self.gamma * expected_Q - self.Q[
                        state[0], state[1], action])
                state = next_state.copy()
                action = next_action

            if done:
                self.timesteps.append(timestep)
                self.episodes.append(episode+1)
                self.timesteps.append(timestep)
                self.episodes.append(episode+2)
                self._update_plot()


algo = SARSA(env, gamma=1)
algo.train(200)

plt.show()