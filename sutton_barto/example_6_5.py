import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from GridWorldv1 import GridWorldEnv

gym.register(
     id="GridWorld-v0",
     entry_point="GridWorldv1:GridWorldEnv",
     max_episode_steps=300,
)

env = gym.make("GridWorld-v0", render_mode = "human")
# env.render_mode = "human"
obs = env.reset()
env.render()

done = False
for i in range(100):
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print('Reward:', obs)
    print('Done:', reward)

# class GridWorldEnv(gym.Env):

#     metadata = {
#         "render_modes": ["human", "rgb_array"],
#         "render_fps": 30,
#     }

#     def __init__(self, render_mode=None, size=(7,10)):
#         self.size = size    # Size of the gridworld
#         self.wind_size = 512 # Screen size

#         # Observations are dictionaries with the agent's and the target's location.
#         # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
#         self.observation_space = spaces.Dict(
#             {
#                 "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#                 "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#             }
#         )
        
#         # We have 4 actions, corresponding to "right", "up", "left", "down"
#         self.action_space = spaces.Discrete(4)

#         """
#         The following dictionary maps abstract actions from `self.action_space` to
#         the direction we will walk in if that action is taken.
#         I.e. 0 corresponds to "right", 1 to "up" etc.
#         """
#         self._action_to_direction = {
#             0: np.array([1, 0]),
#             1: np.array([0, 1]),
#             2: np.array([-1, 0]),
#             3: np.array([0, -1]),
#         }

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

#         """
#         If human-rendering is used, `self.window` will be a reference
#         to the window that we draw to. `self.clock` will be a clock that is used
#         to ensure that the environment is rendered at the correct framerate in
#         human-mode. They will remain `None` until human-mode is used for the
#         first time.
#         """
#         self.window = None
#         self.clock = None   

#     def _get_obs(self):
#         return {"agent": self._agent_location, "target": self._target_location}
    
#     def _get_info(self):
#         return {
#                 "distance": np.linalg.norm(
#                     self._agent_location - self.target_location, ord=1
#                 )
#         }

#     def _agent_escaped(self, location):
#         if np.min(location) < 0 or np.max(location) >= self.size:
#             # print("Agent escaped the gridworld!")
#             return True
#         else:
#             return False
        

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)

#         self._agent_location = np.array([3, 0])
#         self._target_location = np.array([3, 7])

#         observation = self._get_obs()
#         info = self._get_info()

#         if self.render_mode == "human":
#             self._render_frame()

#         return observation, info
    
#     def step(self, action):
#         reward = -1
#         terminated = False

#         move   = self._action_to_direction[action]
#         self._agent_location += move

#         if self._agent_escaped(self._agent_location):
#             # reward = -1
#             self._agent_location = np.clip(self._agent_location, 0, self.size - 1)

#         if np.array_equal(self._agent_location == self._target_location):
#             reward = 1
#             terminated = True

#         observation = self._get_obs()
#         info        = self._get_info()

#         if self.render_mode == "human":
#             self._render_frame()

#         return observation, reward, terminated, False, info

#     def render(self):
#         if self.render_mode == "rgb_array":
#             return self._render_frame()

#     def _render_frame(self):
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode(
#                 (self.window_size, self.window_size)
#             )
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()

#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))
#         pix_square_size = (
#             self.window_size / self.size
#         )  # The size of a single grid square in pixels

#         # First we draw the target
#         pygame.draw.rect(
#             canvas,
#             (255, 0, 0),
#             pygame.Rect(
#                 pix_square_size * self._target_location,
#                 (pix_square_size, pix_square_size),
#             ),
#         )
#         # Now we draw the agent
#         pygame.draw.circle(
#             canvas,
#             (0, 0, 255),
#             (self._agent_location + 0.5) * pix_square_size,
#             pix_square_size / 3,
#         )

#         # Finally, add some gridlines
#         for x in range(self.size + 1):
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (0, pix_square_size * x),
#                 (self.window_size, pix_square_size * x),
#                 width=3,
#             )
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (pix_square_size * x, 0),
#                 (pix_square_size * x, self.window_size),
#                 width=3,
#             )

#         if self.render_mode == "human":
#             # The following line copies our drawings from `canvas` to the visible window
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()

#             # We need to ensure that human-rendering occurs at the predefined framerate.
#             # The following line will automatically add a delay to keep the framerate stable.
#             self.clock.tick(self.metadata["render_fps"])
#         else:  # rgb_array
#             return np.transpose(
#                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
#             )
        

#     def close(self):
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.quit()


# class Windy_Gridworld:

#     def __init__(self):
#         self.world = np.zeros((7, 10))  # World
#         # The world contains in each cell the power of the wind
#         # Positive means it will go up in the real world, i.e. down in the i axis and still in the j axis
#         self.world[:, 3:6] = 1
#         self.world[:, 6:8] = 2
#         self.world[:, 8] = 1
#         self.start_cell = [3, 0]  # Cells where we start
#         self.end_cell = [3, 7]  # Cell where we end
#         self.gamma = 1  # Gamma
#         # Initialize Q matrix: actions are in this order: up - right - down - left (4 actions)
#         self.Q = np.zeros((7, 10, 4))
#         # Initialize policy (will be epsilon-soft)
#         self.epsilon = 0.1
#         self.policy = 0.25 * np.ones((7, 10, 4))
#         # Initialize step size in (0,1]
#         self.alpha = 0.2
#         # Initialize number of episodes
#         self.episodes = 500

#         # Start the Sarsa on-policy TD
#         self.sarsa()

#     def sarsa(self):
#         # Loop for each episode
#         for ep in range(self.episodes):
#             print("START OF EPISODE " + str(ep + 1))
#             # The initial state is fixed, choose initial action based on the epsilon-soft policy
#             current_state = self.start_cell
#             current_action = int(np.random.choice(np.arange(4), 1,
#                                                   p=self.policy[self.start_cell[0]][self.start_cell[1]]))
#             terminal_state = False
#             number_states = 0
#             while not terminal_state:
#                 number_states = number_states + 1
#                 # Take action A, observe R,S'
#                 new_state = [-1, -1]
#                 move_to_make = self.convert_to_move(current_action)
#                 after_move_i = current_state[0]+ move_to_make[0]
#                 # If we go off the grid, I simply put us back in the grid, as if the grid contained walls that
#                 # cannot be crossed.
#                 if after_move_i < 0:
#                     after_move_i = 0
#                 elif after_move_i >= 7:
#                     after_move_i = 6
#                 new_state[0] = current_state[0] + move_to_make[0] - self.world[after_move_i][current_state[1]]
#                 new_state[1] = current_state[1] + move_to_make[1]
#                 if new_state[0] < 0:
#                     new_state[0] = 0
#                 elif new_state[0] >= 7:
#                     new_state[0] = 6
#                 if new_state[1] < 0:
#                     new_state[1] = 0
#                 elif new_state[1] >= 10:
#                     new_state[1] = 9

#                 if new_state[0] - self.end_cell[0] == 0 and new_state[1] - self.end_cell[1] == 0:
#                     current_reward = 1
#                     terminal_state = True
#                 else:
#                     current_reward = -1
#                 new_state[0] = int(new_state[0])
#                 new_state[1] = int(new_state[1])
#                 # Choose A' from S' and the policy
#                 new_action = int(np.random.choice(np.arange(4), 1,p=self.policy[new_state[0]][new_state[1]]))
#                 # Update the Q value
#                 self.Q[current_state[0]][current_state[1]][current_action] = self.Q[current_state[0]][current_state[1]][current_action] \
#                                                                              + self.alpha*(current_reward
#                                                                                            + self.gamma*self.Q[new_state[0]][new_state[1]][new_action]                                                                              - self.Q[current_state[0]][current_state[1]][current_action])
#                 # Update the policy
#                 best_action = 0
#                 best_value = self.Q[current_state[0]][current_state[1]][0]
#                 for a in range(1,4):
#                     current_value = self.Q[current_state[0]][current_state[1]][a]
#                     if current_value > best_value:
#                         best_action = a
#                         best_value = current_value
#                 for a in range(4):
#                     if a == best_action:
#                         self.policy[current_state[0]][current_state[1]][a] = 1 - self.epsilon + (self.epsilon/4)
#                     else:
#                         self.policy[current_state[0]][current_state[1]][a] = self.epsilon/4
#                 # S <-- S' and A <-- A'
#                 current_state = new_state
#                 current_action = new_action
#             print("Number of states: " + str(number_states))

#     def convert_to_move(self, action_number):
#         if action_number == 0:  # Up
#             return [-1, 0]
#         elif action_number == 1:  # Right
#             return [0, 1]
#         elif action_number == 2:  # Down
#             return [1, 0]
#         elif action_number == 3:  # Left
#             return [0, -1]
#         else:  # Problem
#             print("Problem in the action taken - value not supposed to exist")
#             return None

# problem = Windy_Gridworld()