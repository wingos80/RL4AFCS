import gymnasium as gym
import numpy as np
import tiles
import matplotlib.pyplot as plt
import pickle
import timeit
class TileCodingFuncApprox():
    def __init__(self, st_low, st_high, nb_actions, learn_rate, num_tilings, init_val):
        """
        Params:
            st_low      - state space low boundry in all dim, e.g. [-1.2, -0.07] for mountain car
            st_high     - state space high boundry in all dimensions
            nb_actions  - number of possible actions
            learn_rate  - step size, will be adjusted for nb_tilings automatically
            num_tilings - tiling layers - should be power of 2 and at least 4*len(st_low)
            init_val    - initial state-action values
        """
        assert len(st_low) == len(st_high)
        self._n_dim = len(st_low)
        self._lr = learn_rate / num_tilings
        self._num_tilings = num_tilings
        self._scales = self._num_tilings / (st_high - st_low)
        
        # e.g. 8 tilings, 2d space, 3 actions
        # nb_total_tiles = (8+1) * (8+1) * 8 * 3
        nb_total_tiles = (num_tilings+1)**self._n_dim * num_tilings * nb_actions
                
        self._iht = tiles.IHT(nb_total_tiles)
        self._weights = np.zeros(nb_total_tiles) + init_val / num_tilings
        
    def eval(self, state, action):
        assert len(state) == self._n_dim; assert np.isscalar(action)
        scaled_state = np.multiply(self._scales, state)  # scale state to map to tiles correctly
        active_tiles = tiles.tiles(                     # find active tiles
            self._iht, self._num_tilings,
            scaled_state, [action])
        return np.sum(self._weights[active_tiles])       # pick correct weights and sum up

    def train(self, state, action, target):
        assert len(state) == self._n_dim; assert np.isscalar(action); assert np.isscalar(target)

        scaled_state = np.multiply(self._scales, state)  # scale state to map to tiles correctly
        active_tiles = tiles.tiles(                      # find active tiles
            self._iht, self._num_tilings,
            scaled_state, [action])
        value = np.sum(self._weights[active_tiles])      # q-value for state-action pair
        delta = self._lr * (target - value)              # grad is [0,1,0,0,..]
        self._weights[active_tiles] += delta             # ..so we pick active weights instead

class SARSA_N:
    # Episodic Semi-gradient Sarsa with linear function approximator for action-value function
    def __init__(self, env, q_hat, n=4, alpha=0.5, gamma=1, epsilon=0.1, verbose=True, seed=0):
        self.env = env; self.N = env.action_space.n
        self.q_hat = q_hat(env.low, env.high, self.N, alpha, num_tilings=8, init_val=0)
        self.n = n      # number of TD steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.verbose = verbose
        self.seed = seed
        # self._set_up_plot()
    
    def _Q(self, state, action):
        """
        Action-value function

        :param state: current state
        :param action: action to take
        :return: Q value of state-action pair
        """
        Q = self.q_hat.eval(state, action)
        return Q
    
    def _policy(self, state):
        """
        Policy, returns the probablitiy of taking each action

        :param state: current state
        :return: array of probabilities
        """
        N = self.N

        Q = np.zeros(N)
        for i in range(N):
            Q[i] = self._Q(state, i)
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
        p = self._policy(state)
        return np.random.choice(actions, p=p)
    
    def _set_up_plot(self):
        self.fig, self.ax = plt.subplots()
        # set the xlim to between 0 and the max number of episodes
        # self.ax.set_xlim(-10, self.num_episodes+10)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Time step per episode")
        # self.ax.set_title("Currently at Episode 0 (reward=?)")
        self.fig.show()

    def _update_plot(self):    
        self.ax.clear()
        self.ax.plot(self.episodes, self.timesteps)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Time step per episode")
        # self.ax.set_title(f"Currently at Episode {episode} (reward={self.rewards_list[-1]})")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def train(self, max_episodes=100):

        self.timesteps = []
        self.episodes  = []
        for episode in range(max_episodes):
            t  = 0; T = np.inf
            done = False
            seed = self.seed*max_episodes + episode
            np.random.seed(seed)
            obs = self.env.reset(seed=seed)
            states = obs[0]
            action = self._sample_action(states)

            Rt = [0]; St = [states]; At = [action]
            
            done2 = True
            tau = 0
            while True:
                if self.verbose:
                    env.render()
                if t < T:
                    obs, reward, done, _, _ = self.env.step(At[-1])
                    states = obs.copy()
                    # observe and store reward and next state
                    Rt.append(reward)
                    St.append(states)

                    if done & done2:
                        T = t+1
                        done2 = False
                    else:
                        next_action = self._sample_action(states)
                        At.append(next_action)

                tau = t-self.n+1
                
                if tau >= 0:
                    G = 0
                    for i in range(tau+1, min(tau+self.n, T)+1):
                        G += self.gamma**(i-tau-1)*Rt[i-tau-1]
                    if tau+self.n<T: G += self.gamma**(self.n)*self._Q(St[self.n], At[self.n])
                    self.q_hat.train(St[0], At[0], G)

                    if len(At) != 0:
                        debug_action = At[-1]
                    St.pop(0); At.pop(0); Rt.pop(0)

                t += 1 # increment t in the **end**!
                if t % 20 == 0 and self.verbose:
                    print("actino: ", debug_action)
                    print("position: ", St[-1][0], "\n\n")
                    print("time step: ", t)

                if tau == T-1:
                    self.timesteps.append(T)
                    self.episodes.append(episode+1)
                    # self._update_plot()
                    break




env = gym.make("MountainCar-v0", render_mode=None)
obs = env.reset()
q_hat = lambda st_low, st_high, nb_actions, learn_rate, num_tilings, init_val: TileCodingFuncApprox(st_low, st_high, nb_actions, learn_rate, num_tilings, init_val)

num_experiments = 35
max_episodes = 500
times = np.empty((num_experiments,max_episodes))
times_n = np.empty((num_experiments,max_episodes))

# times = {'4': np.zeros((num_experiments,max_episodes))}
times = {'1': np.zeros((num_experiments,max_episodes)),
         '2': np.zeros((num_experiments,max_episodes)),
         '3': np.zeros((num_experiments,max_episodes)),
         '4': np.zeros((num_experiments,max_episodes))}

compute_times = {'1': 0,
                 '2': 0,
                 '3': 0,
                 '4': 0}
for i in range(num_experiments):
    print(f'\nExperiment {i+1}/{num_experiments}')
    for key,value in times.items():
        print(f'    Running {key}-step SARSA')
        tic = timeit.default_timer()
        sarsa_n = SARSA_N(env, q_hat, n=int(key), alpha=0.3, gamma=1, epsilon=0.05, verbose=0, seed=i)
        sarsa_n.train(max_episodes=max_episodes)
        toc = timeit.default_timer()

        value[i,:] = sarsa_n.timesteps
        times[key] = value
        compute_times[key] += toc-tic

        print(f'    Done in {toc-tic:.2f}s')
    

compute_times = {key: value/num_experiments for key,value in compute_times.items()}

h = plt.figure()

for key,value in times.items():
    plt.plot(np.mean(value,axis=0),label=f'{key}-step SARSA, {compute_times[key]:.2f}s')
    # for i in range(num_experiments):
    #     plt.plot(value[i,:],alpha=0.05,color='r',linestyle='--')

plt.yscale('log')
plt.ylim(80,2500)
plt.ylabel('Timesteps per episode')
plt.xlabel('Episode')
plt.legend()
plt.grid()
# plt.show()

plt.savefig('experiment3.png', dpi=300, bbox_inches='tight')

# pickle the times and the figure
with open('times3.pickle', 'wb') as f:
    pickle.dump(times, f)

with open('figure3.pickle', 'wb') as f:
    pickle.dump(h, f)
