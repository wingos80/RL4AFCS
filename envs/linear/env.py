import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import matplotlib.pyplot as plt


class Ce500ShortPeriod(gym.Env):
    def __init__(self, env_config, render_mode=None):
        """
        Gymnasium environment for a cessna citation flight control task. 
        Goal is to control the tracked state to follow the reference trajectory.
        Uses the linear short period citation models for simulation.

        n = state space dimension
        m = action space dimension
        o = observation space dimension

        Important Attributes:
        ----------
        x0: np.ndarray, size=(n, 1)
            Initial state of the system
        dt: float
            Timestep for the simulation
        t_end: float
            End time for the simulation
        reference: dict, len() = 2
            Dictionary containing 'tracked_state', 'reference', where
            'tracked_state' is either "alpha" or "q", and 'reference' is a lambda function 
            that takes time as input and returns the reference value.    
        render_mode: str
            Mode for rendering the environment.        
        """
        
        self._define_coeffs()
        self.A     = self._A()
        self.B     = self._B()
        
        self.C     = np.array([[1, 0],
                               [0, 1]])
        self.D     = np.array([[0],
                               [0]])
        self.x0    = env_config['x0']
        self.x     = self.x0.copy()
        self.dt    = env_config['dt']
        self.t_end = env_config['t_end']
        self.fault_time = env_config['fault_time']
        self.fault_scenario = env_config['fault_scenario']
        self.t     = 0
        self.kappa = 28
        self.stepp = 0
        self.action = 0

        self.tracked_state   = env_config['reference']['tracked_state'][0]
        self.state_reference = env_config['reference']['signal'][0]
        
        self._asserts()

    def _asserts(self):
        assert self.x.shape[0] == self.A.shape[0], f"State vector x0 must have the same size as the state matrix A, got x shape = {self.x.shape} and A shape = {self.A.shape}"
        
        assert self.tracked_state in ["alpha", "q"], f"Tracked state must be either 'alpha' or 'q', got {self.tracked_state}"
        
        # assert callable(self.state_reference), f"Reference state must be a callable function, got {type(self.state_reference)}"
        # assert isinstance(self.state_reference(0), float), f"Reference state function must return a float, got {type(self.state_reference(0))}"

    def _define_coeffs(self):
        
        # PH-LAB short stability and control derivatives 
        # and flight conditions at cruise, see AE3202 lecture notes, table D-1
        self.V    = 59.9       # airspeed, m/s
        self.m    = 4.5478e3   # mass, kg
        self.c    = 2.022      # mean aerodynamic chord, m
        self.S    = 24.2       # wing area, m^2
        self.mu_c = 102.7      # non-dimensional mass, longitudinal (chord)
        self.K2_Y = 0.980 
        self.x_cg = 0.3*self.c      # cg position, m

        # Z = force in the z-direction
        # m = moment along pitch axis
        self.C_Za    = -5.16   # Z coefficient per radian of alpha
        self.C_Zadot = -1.43   # Z coefficient per radian per second of alpha
        self.C_Zq    = -3.86   # Z coefficient per radian per radian per second of alpha
        self.C_Zde   = -0.6238 # Z coefficient per radian of elevator
        self.C_ma    = -0.43   # m coefficient per radian of alpha
        self.C_madot = -3.7    # m coefficient per radian per second of alpha
        self.C_mq    = -7.04   # m coefficient per radian per second of pitch rate
        self.C_mde   = -1.553  # m coefficient per radian of elevator

    def _A(self):
        Vc     = self.V / self.c
        u_cK2Y = self.mu_c * self.K2_Y

        z_a    = Vc * self.C_Za/(2*self.mu_c - self.C_Zadot)
        z_q    =     (2*self.mu_c + self.C_Zq)/(2*self.mu_c - self.C_Zadot)

        m_a    = Vc**2 * (self.C_ma + self.C_Za * self.C_madot/(2*self.mu_c - self.C_Zadot))/(2*u_cK2Y)
        m_q    = Vc * (self.C_mq + self.C_madot * (2*self.mu_c + self.C_Zq)/(2*self.mu_c - self.C_Zadot))/(2*u_cK2Y)

        # sp model only models alpha and q, construct
        # A matrix accordingly using eq 4-47 from AE3202 lecture notes
        A = np.array([[z_a, z_q],
                      [m_a, m_q]])

        # # dimensionalize q by scaling the following coefficients with Vc
        # A[1, 0] *= Vc
        # A[0, 1] /= Vc

        return A
    
    def _B(self):
        Vc = self.V / self.c
        muc_czadot = 2*self.mu_c - self.C_Zadot
        z_de = Vc * (self.C_Zde/muc_czadot)
        m_de = Vc**2*(self.C_mde + self.C_Zde*self.C_madot/muc_czadot)/(2*self.mu_c*self.K2_Y)

        # sp model only has elevator as input
        B = np.array([[z_de],
                      [m_de]])
        return B
    
    def _get_c_grad(self, error_scalar):
        if self.tracked_state == "alpha":
            return self.kappa*np.array([[-2*error_scalar, 0]])
        elif self.tracked_state == "q":
            return self.kappa*np.array([[0, -2*error_scalar]])

    def _engage_fault(self):
        if self.stepp == int(self.fault_time/self.dt):
            case = self.fault_scenario
            if case == 'invert_elevator':    
                # print(f'\n    engaging elevator reversal fault at {self.fault_time} seconds')
                # print(f'\noriginal B matrix\n{self.B}')
                self.B *= -1
                # print(f'\nnew B matrix\n{self.B}')
            elif case == 'damp_elevator':
                # print(f'\n    engaging elevator damp fault at {self.fault_time} seconds')
                self.B *= 0.5
            elif case == 'shift_cg':
                # print(f'\n    engaging cg shift fault at {self.fault_time} seconds')
                # print(f'original A matrix\n{self.A}')
                shift = -0.5
                self.C_mq    += -(self.C_Zq + self.C_ma)*shift/self.c + self.C_Za*(shift/self.c)**2
                self.C_ma    -= self.C_Za*shift/self.c
                self.C_Zq    -= self.C_Za*shift/self.c
                self.C_madot -= self.C_Za*shift/self.c

                # self.C_mde   -= self.C_Zde*shift/self.c
                self.A = self._A()
                # self.B = self._B()
                # print(f'new A matrix\n{self.A}')
            else:
                Exception(f'Fault scenario "{case}" not recognized, exiting...')

            # print(f'\n  engaged fault!!,\n{self.B}')

    def step(self, action: float):
        """
        Simulate the environment for one timestep.
        
        Parameters:
        ----------
        action: float
            Control input to the system in degrees, note do not use tf.tensor here.
        Returns:
        ----------
        obs: np.ndarray, size=(n,1)
            Observation of the system
        reward: float
            Reward for the current time step
        terminated: bool
            Whether the episode is terminated
        info: dict
            Additional information to be returned
        """
        # convert action from deg to rad
        action      = np.deg2rad(action)
        self.action = action
        # Simulate the observations for current timestep
        y   = self.C@self.x + self.D*self.action
        ref = self.state_reference[self.stepp]
        # Calculate reward for current time step
        if self.tracked_state == "alpha":
            error = ref - y[0]
        elif self.tracked_state == "q":
            error = ref - y[1]
        error_scalar = error[0] # Convert to scalar, sometimes it throws a bug cus error is a vector
        reward       = -0.5*self.kappa*error_scalar**2

        reward_grad = self.kappa*np.array([[-2*error_scalar, 0]])

        # Determine state for next time step
        xdot    = self.A@self.x + self.B*self.action
        self.x += xdot*self.dt

        # Update time
        self.t += self.dt
        
        self.stepp += 1
        self._engage_fault()
        # if self.stepp == 1000:
        #     self.B *= -1
        #     print(f'\n  engaged fault!!,\n{self.B}')

        
        # Check if episode is terminated
        terminated = self.t >= self.t_end

        info = {       'yref':    ref,
                          't': self.t,
                          'x': self.x,
                          'e': error_scalar,
                'reward_grad': reward_grad} # Any additional information to be returned

        obs = self.x

        # Store history
        self.x_hist.append(self.x)
        self.y_hist.append(y)
        self.yref_hist.append(ref)
        return obs, reward, terminated, False, info

    def reset(self,seed=None):
        """
        Reset the environment to initial state.
        Parameters:
        ----------
        seed: int
            Seed for the random number generator
        Returns:
        ----------
        obs: np.ndarray, size=(o,1)
            Empty observation of the system
        """
        super().reset(seed=seed)
        self.x = self.x0.copy()
        self.t = 0
        self.stepp = 0

        # reset the dynamics
        self._define_coeffs()
        self.A     = self._A()
        self.B     = self._B()
        
        self.x_hist = []
        self.y_hist = []
        self.yref_hist = []
        

        # obs = np.zeros(1)
        obs = self.x
        reward = 0
        terminated = False
        info = {          'yref': self.state_reference[self.stepp],
                             't': 0,
                             'x': self.x,
                             'e': 0,
                   'reward_grad': self._get_c_grad(0)}
        return obs, reward, terminated, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

class FlyingVShortPeriod(gym.Env):
    def __init__(self, env_config, render_mode=None):
        """
        Gymnasium environment for a Flying-V flight control task. 
        Goal is to control the tracked state to follow the reference trajectory.

        n = state space dimension
        m = action space dimension
        o = observation space dimension

        Parameters:
        ----------
        x0: np.ndarray, size=(n, 1)
            Initial state of the system
        dt: float
            Timestep for the simulation
        t_end: float
            End time for the simulation
        reference: dict, len() = 2
            Dictionary containing 'tracked_state', 'reference', where
            'tracked_state' is either "alpha" or "q", and 'reference' is a lambda function 
            that takes time as input and returns the reference value.    
        render_mode: str
            Mode for rendering the environment.        
        """
        self.A     = np.array([[-0.4021, 0.9913],
              [-3.806, -0.4977]])
        self.B     = np.array([[0.0008416],
              [0.02884]])
        self.C     = np.array([[1, 0],
                               [0, 1]])
        self.D     = np.array([[0],
                               [0]])
        self.x0    = env_config['x0']
        self.x     = self.x0.copy()
        self.dt    = env_config['dt']
        self.t_end = env_config['t_end']
        self.t     = 0
        self.stepp  = 0
        self.kappa = 28
        

        self.tracked_state   = env_config['reference']['tracked_state']
        self.state_reference = env_config['reference']['reference']

        self._asserts()

    def _asserts(self):
        assert self.x.shape[0] == self.A.shape[0], f"State vector x0 must have the same size as the state matrix A, got x shape = {self.x.shape} and A shape = {self.A.shape}"
        
        assert self.tracked_state in ["alpha", "q"], f"Tracked state must be either 'alpha' or 'q', got {self.tracked_state}"
        
        assert callable(self.state_reference), f"Reference state must be a callable function, got {type(self.state_reference)}"
        assert isinstance(self.state_reference(0), float), f"Reference state function must return a float, got {type(self.state_reference(0))}"

    def _get_c_grad(self, error_scalar):
        if self.tracked_state == "alpha":
            return self.kappa*np.array([[-2*error_scalar, 0]])
        elif self.tracked_state == "q":
            return self.kappa*np.array([[0, -2*error_scalar]])
        
    def step(self, action: float):
        """
        Simulate the environment for one timestep.
        
        Parameters:
        ----------
        action: float
            Control input to the system, note do not use tf.tensor here.
        Returns:
        ----------
        obs: np.ndarray, size=(n,1)
            Observation of the system
        reward: float
            Reward for the current time step
        terminated: bool
            Whether the episode is terminated
        info: dict
            Additional information to be returned
        """
        # Simulate the observations for current timestep
        y = self.C@self.x + self.D*action
        ref = self.state_reference(self.t)
        # Calculate reward for current time step
        if self.tracked_state == "alpha":
            error = ref - y[0]
        elif self.tracked_state == "q":
            error = ref - y[1]
        error_scalar = error[0] # Convert to scalar, sometimes it throws a bug cus error is a vector
        reward = -0.5*self.kappa*error_scalar**2

        reward_grad = self.kappa*np.array([[-2*error_scalar, 0]])

        # Determine state for next time step
        xdot    = self.A@self.x + self.B*action
        self.x += xdot*self.dt

        # Update time
        self.t += self.dt

        self.stepp += 1
        if self.stepp == 1000:
            self.B *= -1
            # print(f'\n  engaged fault!!,\n{self.B}')
        
        # Check if episode is terminated
        terminated = self.t >= self.t_end

        info = {       'yref':    ref,
                          't': self.t,
                          'x': self.x,
                          'e': error_scalar,
                'reward_grad': reward_grad} # Any additional information to be returned

        obs = self.x
        # obs = error

        # Store history
        self.x_hist.append(self.x)
        self.y_hist.append(y)
        self.yref_hist.append(ref)
        return obs, reward, terminated, False, info

    def reset(self,seed=None):
        """
        Reset the environment to initial state.
        Parameters:
        ----------
        seed: int
            Seed for the random number generator
        Returns:
        ----------
        obs: np.ndarray, size=(o,1)
            Empty observation of the system
        """
        super().reset(seed=seed)
        self.x = self.x0.copy()
        self.t = 0
        
        self.x_hist    = []
        self.y_hist    = []
        self.yref_hist = []
        
        self.A     = A.copy()
        self.B     = B.copy()
        self.C     = C.copy()
        self.D     = D.copy()
        self.stepp = 0  
        # obs = np.zeros(1)
        obs        = self.x
        reward     = 0
        terminated = False
        info       = {       'yref': self.state_reference(self.t),
                                't': 0,
                                'x': self.x,
                                'e': 0,
                      'reward_grad': self._get_c_grad(0)}
        return obs, reward, terminated, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
