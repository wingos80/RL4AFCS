import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
# make citation import agnostic to where the file is called
if __package__ or "." in __name__:
    from .extended_input import citation as c_model
else:
    import extended_input.citation as c_model

class Ce500NonLinear(gym.Env):
    def __init__(self, env_config, render_mode=None):
        """
        Gymnasium environment for a cessna citation flight control task. 
        Goal is to control the tracked state to follow the reference trajectory.
        Uses the nonlinear citation models for simulation.

        n = state space dimension
        m = action space dimension
        o = observation space dimension

        Model state and inputs:
            Simulated States: p, q, r, vtas, alpha, beta, phi, theta, psi, he, xe, ye
            Model States    : alpha, theta, q, phi, beta, p, r
            MDP States      : alpha, theta, q, theta - theta_ref
            Inputs          : de, da, dr, trim de, trim da, trim dr, df, gear, throttle1, throttle2
        """

        self.fault_scenario = env_config['fault_scenario']
        
        self.model       = c_model
        self.initialized = False

        # No need to define initial condition, already built into .pyd
        # self.x0    = env_config['x0'] 

        # miscellaneous model parameters
        self.dt         = env_config['dt']
        self.t_end      = env_config['t_end']
        self.total_steps= env_config['total_steps']
        self.fault_time = env_config['fault_time']
        self.trim_state = env_config['trim_state']
        self.trim_input = env_config['trim_input']

        # state and action space dimensions
        self.state  = self.trim_state.copy()
        self.mdp_s_dim = env_config['state_dim']
        self.t      = 0
        self.kappa  = [1, 1, 1] # reward scaling/weight for p, q, r respectively
        self.stepp  = 0
        self.action = None

        self.tracked_state   = env_config['reference']['tracked_state']
        self.state_reference = env_config['reference']['signal']

        self._set_saturations()
        self._set_surfaces_dynamics()
        self._set_weight_matrices(self.kappa)

    def _set_weight_matrices(self, kappa):
        """
        Define the weight matrices for the cost function
        """
        self.Q_sym = kappa[1]
        self.Q_asym = np.diag([kappa[0], kappa[2]])

    def _reset_surfaces_states(self):
        """
        Reset the actuator dynamics states
        """
        self.x0_act = np.zeros((3,1))
        # self.x0_act[0,0] = self.trim_input[0]

    def _set_surfaces_dynamics(self, omega_0=13, reset=True):
        """
        Define the actuator dynamics state space model
        x = [de, da, dr]
        u = [de_cmd, da_cmd, dr_cmd]
        """
        # actuator dynamics state space model:
        c_act_a = np.array([[-omega_0,   0,   0],
                            [  0, -omega_0,   0],
                            [  0,   0, -omega_0]])

        c_act_b = np.array([[omega_0,  0, 0],
                            [ 0, omega_0, 0],
                            [ 0,  0, omega_0]])
        
        # simple forward euler integration scheme for discretization
        self.act_A_inc = np.eye(3) + self.dt*c_act_a
        self.act_B_inc = self.dt*c_act_b

        self.act_A_dt = c_act_a
        self.act_B_dt = c_act_b
        self.act_omega_0 = omega_0
        if reset:
            self._reset_surfaces_states()

    def _set_saturations(self):
        """
        Define control surfaces saturation limits in degrees
        """
        # asymmetric limits for elevator, more realistic but not used 
        # self.limits = {'de': np.array([-20, 15]),
        #                'da': np.array([-37, 37]),
        #                'dr': np.array([-22, 22])}
        self.limits = {'de': np.array([-15, 15]),
                       'da': np.array([-37, 37]),
                       'dr': np.array([-22, 22])}
                
    def _scale_action(self, action):
        """
        Scale action from [-1, 1] to control surface limits
        """
        # scaling from -1 +1 to -saturation +satutaion
        action[0] = action[0] * (self.limits['de'][1] - self.limits['de'][0])/2 
        action[1] = action[1] * (self.limits['da'][1] - self.limits['da'][0])/2 
        action[2] = action[2] * (self.limits['dr'][1] - self.limits['dr'][0])/2 

        # shifting the action to be centered around the saturation limits
        action[0] = action[0] + (self.limits['de'][1] + self.limits['de'][0])/2
        action[1] = action[1] + (self.limits['da'][1] + self.limits['da'][0])/2
        action[2] = action[2] + (self.limits['dr'][1] + self.limits['dr'][0])/2
        return np.deg2rad(action)
    
    def _get_c_grad(self, error_scalar):
        pass

    def _engage_fault(self, surface_deflections):
        effective_input = np.zeros(11)
        effective_input[:3] = surface_deflections.copy()
        if self.stepp >= int(self.fault_time/self.dt):
            case = self.fault_scenario
            if 'damp_elevator' in case:
                effective_input[0] *= 0.3
            elif 'damp_aileron' in case:
                effective_input[1] *= 0.3
            elif 'damp_rudder' in case:
                effective_input[2] *= 0.3
            elif 'damp_all' in case:
                effective_input[:3] *= 0.3
            elif 'shift_cg' in case:
                effective_input[-1] = -0.5
            elif 'slow_all' in case:
                self._set_surfaces_dynamics(6, reset=False)
            else:
                Exception(f'Fault scenario "{case}" not recognized, exiting...')
        return effective_input
    
    def _saturate_surfaces(self, surface_deflections):
        if self.stepp >= int(self.fault_time/self.dt):
            case = self.fault_scenario
            if 'saturate_elevator' in case:
                surface_deflections[0] = np.clip(surface_deflections[0], -np.deg2rad(5), np.deg2rad(5))
            elif 'saturate_aileron' in case:
                surface_deflections[1] = np.clip(surface_deflections[1], -np.deg2rad(18), np.deg2rad(18))
            elif 'saturate_rudder' in case:
                surface_deflections[2] = np.clip(surface_deflections[2], -np.deg2rad(10), np.deg2rad(10))
        return surface_deflections
    
    def _propagate_surfaces_states(self, action):
        """
        Propagate the actuator dynamics
        """
        cmd = action.copy()
        cmd = cmd.reshape(-1,1)
        # self.x0_act = self.act_A_inc@self.x0_act + self.act_B_inc@cmd # no rate saturation version

        # # rate saturate version 1
        # x_dot = self.act_A_dt@self.x0_act + self.act_B_dt@cmd
        # x_dot = np.clip(x_dot, -19.7, 19.7)
        # self.x0_act += self.dt*x_dot

        # rate saturation version 2
        x_diff = cmd - self.x0_act
        x_diff *=self.act_omega_0
        x_diff = np.clip(x_diff, -np.deg2rad(19.7), np.deg2rad(19.7))
        self.x0_act += self.dt*x_diff

        return self.x0_act.copy().flatten()

    def step(self, action: np.array):
        """
        Simulate the environment for one timestep.
        
        Parameters:
        ----------
        action: np.array, size=(m,)
            Control surfaces input to the system in, note do not use tf.tensor here,
            action should be normalized values between [-1, 1]
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
        
        # scale action to aileron, elevator, and rudder deflection in degrees
        action              = self._scale_action(action)
        surface_deflections = self._propagate_surfaces_states(action)
        surface_deflections = self._saturate_surfaces(surface_deflections)
        model_input         = self.trim_input.copy()
        model_input        += self._engage_fault(surface_deflections)

        self.state       = self.model.step(model_input)
        attitude_rates   = self.state[6:9].copy() # phi, theta, psi
        ref_signals      = [ref_func[self.stepp] for ref_func in self.state_reference] # p_ref, q_ref, r_ref

        # calculate the reward for the symmetric and asymmetric MDPs
        error           = np.array([state - ref for ref, state in zip(ref_signals, attitude_rates)])

        error_lon       = error[1]
        reward_lon      = np.array([[-0.5*self.Q_sym*error_lon**2]])
        reward_grad_lon = np.zeros((1,3))
        reward_grad_lon[0,-1] = -self.Q_sym*error_lon

        error_lat       = np.array([[error[0]], 
                                     [error[2]]])
        reward_lat      = -0.5*error_lat.T@self.Q_asym@error_lat
        reward_grad_lat = -self.Q_asym@error_lat
        reward_grad_lat = reward_grad_lat.T # transpossing to return a row vector as reward grad

        self.stepp += 1
        self.t     += self.dt
        
        lon_state_indx = [4, 7, 1]
        lat_state_indx = [6, 5, 0, 2]
        x_lon, x_lat   = self.state[lon_state_indx], self.state[lat_state_indx]
        zeros          = np.zeros((1,2))
        
        MDP_state     = np.zeros(self.mdp_s_dim)
        MDP_state[:3] = x_lon
        MDP_state[-1] = error[1]

        # return the state, reward, and termination signal
        info = {      'nans' : True if np.isnan(self.state).any() else False,
                         's' : MDP_state,
                      'yref' : ref_signals,
           'action_commanded': surface_deflections,
           'action_effective': model_input[:3],
                     'rates' : attitude_rates,
                         't' : self.t,
                    'x_full' : self.state,
                         'x' : [x_lon.reshape(-1,1), x_lat.reshape(-1,1)],
                         'e' : error, 
                        'RSE': [np.sqrt(error[1]**2), np.sqrt(error[0]**2 + error[2]**2)], # angles and angular rates in radians
                'reward_grad': [reward_grad_lon, 
                                np.reshape(np.concatenate((zeros, reward_grad_lat), axis=1), (1,-1))]}
        
        reward    = np.array([reward_lon, reward_lat])
        return MDP_state, reward, None, False, info
    
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
        reward: np.ndarray, size=(2,)
            Empty reward for the current time step
        terminated: bool
            Whether the episode is terminated
        info: dict
            Additional information to be returned
        """
        super().reset(seed=seed)

        if self.initialized:
            # reinitialize the model if already previously initialized
            self.model.terminate()
            self.model.initialize()
        else:
            # initialize the model if not already initialized
            self.initialized = True
            self.model.initialize()

        # run the airplane for 10 seconds to get it into a trim state
        for _ in range(int(10/self.dt)):
            self.model.step(self.trim_input)
        
        self.state = self.model.step(self.trim_input)

        MDP_state = np.zeros(self.mdp_s_dim)
        self.stepp = 0
        self.t     = 0

        self._reset_surfaces_states()

        info = {      'nans' : False,
                         's' : MDP_state,
                      'yref' : np.zeros(3),
                     'action': np.zeros(3),
                     'rates' : np.zeros(3),
                         't' : self.t,
                    'x_full' : self.state,
                         'x' : [np.zeros((3,1)), np.zeros((4,1))],
                         'e' : np.zeros(3), 
                        'RSE': [0, 0], # angles and angular rates in radians
                'reward_grad': [np.zeros(2),
                                np.zeros(4)]}
        return MDP_state, np.zeros((2,1,1)), None, None, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.model.terminate()
        self.initialized = False



if __name__ == '__main__':
    states = []
    refs   = [] 
    errors = []
    ctrl   = []
    dt     = 0.01
    t_end  = 1*60
    total_steps = int(t_end/dt)
    times = np.linspace(0,t_end,total_steps)

    p_ref = np.zeros(total_steps)
    q_ref = np.zeros(total_steps)
    r_ref = np.zeros(total_steps)

    # making sine reference roll rates between 10 and 40 seconds
    p_ref[1000:5000] = np.deg2rad(5)*-np.sin(2*np.pi*np.linspace(0,40,4000)/20)
    # p_ref[3500:4500] = np.deg2rad(5)*np.sin(2*np.pi*np.linspace(0,10,1000)/20)

    # # # making sine reference roll rates between 10 and 40 seconds
    # p_ref[1500:2000] = np.deg2rad(3)*np.linspace(0, 5, 500)/5
    # p_ref[2000:2500] = np.deg2rad(3)
    # p_ref[2500:3000] = np.deg2rad(3)*np.linspace(5, 0, 500)/5
    # # # making sine reference roll rates between 10 and 40 seconds
    # p_ref[3000:3500] = -np.deg2rad(3)*np.linspace(0, 5, 500)/5
    # p_ref[3500:4000] = -np.deg2rad(3)
    # p_ref[4000:4500] = -np.deg2rad(3)*np.linspace(5, 0, 500)/5

    # making a sine reference pitch rate for the first 10 seconds and the 45th to 55th seconds
    q_ref[:1000] = np.deg2rad(3)*np.sin(2*np.pi*np.linspace(0,10,1000)/10)
    q_ref[4500:5500] = np.deg2rad(3)*-np.sin(2*np.pi*np.linspace(0,10,1000)/10)
    # t1 = 300
    # t2 = t1+500
    # t3 = t2+300

    # q_ref[0:t1] = np.deg2rad(1)*np.linspace(0, 3, 300)
    # q_ref[t1:t2] = np.deg2rad(1)*3
    # q_ref[t2:t3] = np.deg2rad(1)*np.linspace(3, 0, 300)
    
    # t0 = t3+500
    # t1 = t0+300
    # t2 = t1+500
    # t3 = t2+300
    # q_ref[t0:t1] = np.deg2rad(1)*np.linspace(0, -3, 300)
    # q_ref[t1:t2] = -np.deg2rad(1)*3
    # q_ref[t2:t3] = np.deg2rad(1)*np.linspace(-3, 0, 300)
    # # Inputs: de, da, dr, trim de, trim da, trim dr, df, gear, throttle1, throttle2
    # # States: p, q, r, vtas, alpha, beta, phi, theta, psi, he, xe, ye
    # trim_input = np.array([-0.02855, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.55])
    # trim_state = np.array([0, 0, 0, 90, 0.0576, 0, 0, 0.0576, 0, 2000, 0, 0])
    # env_config = {'state_dim' : 2,
    #               'action_dim': 3,
    #               'trim_input': trim_input,
    #               'trim_state': trim_state,
    #               'dt'        : dt,
    #               't_end'     : t_end,
    #               'fault_time': 20,
    #               'fault_scenario': 'shift_cg',
    #               'fault_scenario': 'none',
    #               'reference'     : {'tracked_state' : ['p', 'q', 'r'],
    #                                  'signal'        : [p_ref, q_ref, r_ref]}
    #              }
    # citation = Ce500NonLinear(env_config)
    # citation.reset()

    # for i in range(total_steps):
    #     elevator = 0.1*np.random.rand()
    #     aileron  = np.random.rand()
    #     rudder   = np.random.rand()

        
    #     _,_,_,_, info = citation.step(np.array([0, 0, 0]))
    #     states.append(info['state'])
    #     errors.append(info['error'])
    #     refs.append(info['reference'])

        
    # citation.close()

    # states = np.array(states)
    # np.savetxt("python_states.csv", states, delimiter=",")
    # refs   = np.array(refs)
    # errors = np.array(errors)

    # plt.figure()
    # plt.plot(times, refs[:,0]  , label='p_ref')
    # plt.plot(times, states[:,0], label='p')
    # plt.plot(times, errors[:,0], label='p_error')
    # plt.legend()
    # plt.grid()
    # plt.ylabel(r'$p$ [rad/s]')
    # plt.xlabel('Time [s]')
    # plt.subplots_adjust(left=0.155, top=0.96,bottom=0.14, right=0.98)

    # plt.figure()
    # plt.plot(times,states[:,-3])
    # plt.ylabel('Altitude [m]')
    # plt.xlabel('Time [s]')  
    # plt.subplots_adjust(left=0.155, top=0.96,bottom=0.14, right=0.98)
    # plt.grid()

    # plt.figure()
    # plt.plot(times,states[:,7])
    # plt.ylabel(r'$\theta$ [rad]')
    # plt.xlabel('Time [s]')  
    # plt.subplots_adjust(left=0.155, top=0.96,bottom=0.14, right=0.98)
    # plt.grid()

    # plt.figure()
    # plt.plot(times,states[:,1])
    # plt.ylabel(r'$q$ [rad/s]')
    # plt.xlabel('Time [s]')
    # plt.subplots_adjust(left=0.155, top=0.96,bottom=0.14, right=0.98)
    # plt.grid()

    # plt.figure()
    # plt.plot(times,states[:,0])
    # plt.ylabel(r'$p$ [rad/s]')
    # plt.xlabel('Time [s]')
    # plt.subplots_adjust(left=0.155, top=0.96,bottom=0.14, right=0.98)
    # plt.grid()

    # plt.figure()
    # plt.plot(times,states[:,2])
    # plt.ylabel(r'$r$ [rad/s]')
    # plt.xlabel('Time [s]')
    # plt.subplots_adjust(left=0.155, top=0.96,bottom=0.14, right=0.98)
    # plt.grid()

    # plt.figure()
    # plt.plot(times,states[:,3])
    # plt.ylabel(r'$V_{TAS}$ [rad/s]')
    # plt.xlabel('Time [s]')
    # plt.subplots_adjust(left=0.155, top=0.96,bottom=0.14, right=0.98)
    # plt.grid()



    # integrate p_ref to get phi_ref, and integrate q_ref to get theta_ref
    phi_ref = np.zeros(total_steps)
    theta_ref = np.zeros(total_steps)
    for i in range(1, total_steps):
        phi_ref[i] = phi_ref[i-1] + p_ref[i]*dt
        theta_ref[i] = theta_ref[i-1] + q_ref[i]*dt
    
    theta_ref += np.deg2rad(0.0576) #adding the trim pitch to theta ref


    # plotting, convert all rates and angles to degrees first
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(times,np.rad2deg(p_ref))
    ax[0].set_ylabel(r'$p$ [deg/s]')
    ax[0].grid()

    ax[1].plot(times,np.rad2deg(phi_ref))
    ax[1].set_ylabel(r'$\phi$ [deg]')
    ax[1].set_xlabel('Time (s)')
    ax[1].grid()

    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(times,np.rad2deg(q_ref))
    ax[0].set_ylabel(r'$q$ [deg/s]')
    ax[0].grid()

    ax[1].plot(times,np.rad2deg(theta_ref))
    ax[1].set_ylabel(r'$\theta$ [deg]')
    ax[1].set_xlabel('Time (s)')
    ax[1].grid()

    plt.show()
    print('done')