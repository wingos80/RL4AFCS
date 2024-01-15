import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, render_mode=None, size=np.array([7,10])):
        self.size = size    # Size of the gridworld
        self.window_size = size*100 # Screen size

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0]), high=np.array([size[0], size[1]]), dtype=int),
                "target": spaces.Box(low=np.array([0, 0]), high=np.array([size[0], size[1]]), dtype=int),
            }
        )
        
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None   

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
                "distance": np.linalg.norm(
                    self._agent_location - self._target_location, ord=1
                )
        }

    def _check_agent_escaped(self, location):
        location2 = np.clip(location, [0,0], self.size - 1)
        
        if np.array_equal(location, location2):
            return False, location2
        else:
            return True, location2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([3, 0])
        self._target_location = np.array([3, 7])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        reward = -1
        terminated = False

        move   = self._action_to_direction[action]
        self._agent_location += move

        _, self._agent_location = self._check_agent_escaped(self._agent_location)

        if np.array_equal(self._agent_location, self._target_location):
            reward = 0
            terminated = True

        observation = self._get_obs()
        info        = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size[1], self.window_size[0])
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size[1], self.window_size[0]))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size[0] / self.size[0]
        )  # The size of a single grid square in pixels

        def _real_to_screen(self, real_location):
            l1 = np.flip(real_location.copy())
            l1[1] = self.size[0] - l1[1]
            return l1*pix_square_size

        _os_target_location = _real_to_screen(self, self._target_location + np.array([1, 0]))
        _os_agent_location  = _real_to_screen(self, self._agent_location + 0.5)
        
        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                _os_target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            _os_agent_location,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size[1], pix_square_size * x),
                width=3,
            )
        
        for y in range(self.size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * y, 0),
                (pix_square_size * y, self.window_size[0]),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    