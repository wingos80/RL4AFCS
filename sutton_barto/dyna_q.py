import numpy as np
import pygame
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from GridWorldv1 import GridWorldEnv


class model:
    def __init__(self):
        """
        Tabular model of a discrete MDP, 
        stores the reward and next state for each action taken in each state"""
        self.memory = {}


    def add_experience(self, s, a, r, s_next):
        """
        Add a new experience to the model
        
        Parameters
        ----------
        s : (tuple/array like)
            current state            
        a : (int)
            action taken        
        r : (float)
            reward received        
        s_next : (tuple/array like)
            next state
            
        Returns
        -------
        None
        """
        state = (s[0], s[1])
        if state not in self.memory:
            self.memory[state] = {}
        self.memory[state][a] = (r, s_next)

    def sample_random(self):
        """
        Sample a completely random experience from the model

        Parameters
        ----------
        None
        
        Returns
        -------
        (s, a, r, s_next) : (tuple)
            reward and next state
        """
        id_rand = np.random.randint(len(self.memory.keys()))
        all_keys = list(self.memory.keys())
        s1, s2 = all_keys[id_rand]
        s = np.array([s1, s2])
        actions_taken = list(self.memory[(s1, s2)].keys())

        # sample a random action
        a = actions_taken[np.random.randint(len(actions_taken))]

        r, s_next = self.sample(s, a)
        return s, a, r, s_next

    def sample(self, s, a):
        """
        Sample a random experience from the model given a state and action

        Parameters
        ----------
        s : (tuple)
            current state   
        a : (int)
            action taken
        
        Returns
        -------
        (r, s_next) : (tuple)
            reward and next state
        """

        state = (s[0], s[1])
        assert state in self.memory, "Sampled from empty memory"

        sample = self.memory[state][a]
        return sample
    
class DynaQ:
    def __init__(self, env, ps=5, alpha=0.5, gamma=1, epsilon=0.1, verbose=True, seed=0):
        """
        Dyna-Q algorithm
        
        Parameters
        ----------
        env : (gym environment)
            environment to train on
        ps : (int)
            number of planning steps
        alpha : (float)
            learning rate
        gamma : (float)
            discount factor
        epsilon : (float)
            exploration rate
        
        Returns
        -------
        None
        """
        self.env = env
        self.ps = ps
        print("ps", ps)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = env.action_space.n
        high = env.observation_space['agent'].high
        low = env.observation_space['agent'].low
        steps = high - low + 1
        self.Q = np.zeros((steps[0], steps[1], self.N))

        self.verbose = verbose
        self.seed = seed


    def _policy(self, state):
        """
        Policy, returns the probablitiy of taking each action using an epsilon greedy method

        Parameters
        ----------
        state : (tuple)
            current state
        
        Returns
        -------
        p : (array)
            probability distribution over actions
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
        
        Parameters
        ----------
        state : (tuple)
            current state

        Returns
        -------
        action : (int)
            action to take
        """
        actions = np.arange(self.N)
        return np.random.choice(actions, p=self._policy(state))
        # return 1
    
    def _Q_eps_geedy(self, epsilon, state):
        """
        Epsilon greedy policy, picks the action with the highest Q value with 1- epsilon probability
        and a random action with epsilon probability
        
        Parameters
        ----------
        epsilon : (float)
            exploration rate
        state : (tuple)
            current state
        
        Returns
        -------
        action : (int)
            action to take
        """
        Q = self.Q[state[0], state[1]]
        if np.random.random() < epsilon:
            return np.random.randint(0, self.N)
        else:
            return np.random.choice(np.flatnonzero(Q == Q.max()))

    def _set_up_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.title.set_text("Dyna-Q timesteps per episode")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Time steps per episode")
        self.fig.show()

        self.fig2, self.ax2 = plt.subplots()
        self.ax2.title.set_text("Dyna-Q Vmap")
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("y")
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(0, 9)
        self.ax2.set_ylim(0, 6)
        Vmap = self.create_Vmap()
        cm = self.ax2.pcolormesh(Vmap)
        self.cbar = self.fig2.colorbar(cm, ax=self.ax2)

        self.fig2.show()
        
    def _update_plot(self):    
        self.ax.clear()
        self.ax.plot(self.episodes, self.timesteps)
        self.ax.set_yscale('log')
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Time steps per episode")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_Vmap(self):
        self.ax2.clear()
        Vmap = self.create_Vmap()
        self.ax2.pcolormesh(Vmap)
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("y")
        self.ax2.set_title("Vmap")
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(0, 9)
        self.ax2.set_ylim(0, 6)
        self.cbar.vmin=Vmap.min()
        self.cbar.vmax=Vmap.max()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

    def create_Vmap(self):

        Vmap = np.sum(self.Q,axis=2)

        return Vmap

    def train(self, max_episodes=100):
        """
        Train the agent for the given number of episodes

        Parameters
        ----------
        episodes : (int)
            number of episodes to train

        Returns
        -------
        None
        """
        if self.verbose:
            self._set_up_plot()

        self.timesteps = []
        self.episodes  = []
        self.model = model()
        for episode in range(max_episodes):
            timestep  = 0
            done = False

            seed = self.seed*max_episodes + episode
            np.random.seed(seed)
            obs, _ = self.env.reset(seed=seed)

            state = obs['agent'].copy()
            action = self._Q_eps_geedy(0, state)
            while not done:
                # self.env.render()
                timestep += 1
                obs, reward, done, _, _ = self.env.step(action)
                next_state = obs['agent'].copy()
                # next_action = self._Q_eps_geedy(self.epsilon, next_state)
                next_action = self._sample_action(next_state)
                expected_Q = np.sum(self._policy(next_state) * self.Q[next_state[0], next_state[1]])
                SARSA_Q = self.Q[next_state[0], next_state[1], next_action]
                # if not done:
                learning_Q = self.Q[next_state[0], next_state[1], self._Q_eps_geedy(0, next_state)]
                # else:
                #     learning_Q = self.Q[next_state[0], next_state[1], self._Q_eps_geedy(0, next_state)]
                self.Q[state[0], state[1], action] += self.alpha * (
                            reward + self.gamma * learning_Q - self.Q[state[0], state[1], action])
                
                self.model.add_experience(state, action, reward, next_state)

                state = next_state.copy()
                action = next_action


                for _ in range(self.ps):
                    s, a, r, sp = self.model.sample_random()

                    # # perform q update from simulated trajectories with your choice of q learning algorithm
                    # learning_Q = self.Q[sp[0], sp[1], self._Q_eps_geedy(0, sp)]

                    if np.array_equal(sp, np.array([5, 8])):
                        learning_Q = 0
                    else:
                        learning_Q = self.Q[sp[0], sp[1], self._Q_eps_geedy(0, sp)]

                    self.Q[s[0], s[1], a] += self.alpha * (
                                r + self.gamma * learning_Q - self.Q[s[0], s[1], a])
                
                self._update_Vmap()

            if done:
                self.timesteps.append(timestep)
                self.episodes.append(episode+1)
                if self.verbose:
                    print("episode: ", episode)
                    print("timesteps: ", timestep)
                    print("Q: ", self.Q)
                    print("model: ", self.model.memory)
                    print("")
                    self._update_plot()
                    self._update_Vmap()


# np.random.seed(42)

gym.register(
     id="GridWorld-v1",
     entry_point="GridWorldv1:GridWorldEnv",
     max_episode_steps=300,
)

env = gym.make("GridWorld-v1",render_mode='human')
obs = env.reset()

all_timesteps = []
all_episodes = []

for i in range(1):
    print("run: ", i)
    algo = DynaQ(env, ps=10, gamma=0.95, alpha=0.1, seed=i,verbose=1)
    algo.train(200)

    all_timesteps.append(algo.timesteps)
    all_episodes.append(algo.episodes)


plt.show()
# # take average timesteps used
# timesteps = np.mean(all_timesteps, axis=0)
# plt.plot(all_episodes[0], timesteps)
# plt.yscale('log')
# plt.xlabel("Episode")
# plt.ylabel("Time steps per episode")
# plt.grid()
# plt.ylim(10**1, 5*10**2)
# plt.show()