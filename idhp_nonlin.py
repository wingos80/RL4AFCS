"""
MIT License

Copyright (c) 2024 wingos80

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
from objects import IDHPnonlin
from envs.nonlinear.env import Ce500NonLinear
from functions import *


if __name__ == '__main__':
    tic         = timeit.default_timer()
    t_end, dt   = 90, 0.01                         # flight time and time step in seconds
    total_steps = int(t_end/dt)                    # total number of steps
    times       = np.linspace(0,t_end,total_steps) # time vector

    theta_ref = 0.0576+np.zeros(total_steps)

    # making a sine reference pitch rate for the first 10 seconds and the 45th to 55th seconds
    theta_ref[:4500] += np.deg2rad(5)*np.sin(2*np.pi*np.linspace(0,45,4500)/15)*(np.linspace(2.0,0.8,4500))
    theta_ref[:4500] += np.deg2rad(4)*np.sin(2*np.pi*np.linspace(0,45,4500)/30)*(np.linspace(2.0,0.8,4500))

    theta_ref[5500:6500] += np.deg2rad(1.5)*np.linspace(0,10,1000)
    theta_ref[6500:7500] += np.deg2rad(15)
    theta_ref[7500:8500] += np.deg2rad(1.5)*np.linspace(10,0,1000)

    # Begin specifying environment
    # Inputs: de, da, dr, trim de, trim da, trim dr, df, gear, throttle1, throttle2, xcg
    # States: p, q, r, vtas, alpha, beta, phi, theta, psi, he, xe, ye
    trim_input = np.array([-0.02855, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.55, 0])
    trim_state = np.array([0, 0, 0, 90, 0.0576, 0, 0, 0.0576, 0, 2000, 0, 0])

    # 1 for Monte Carlo simulations, 0 for single run
    MC = 0
    if MC:
        # Monte Carlo run
        tic = timeit.default_timer()

        _, etaah, etaal, etach, etacl, lambda_hs, lambda_ls = get_all_used_hparams()

        # good collections 2 no rate saturation
        configs = {'etaah': [40, 25, 43, 30],
                   'etaal': [10, 7.5, 6.5, 5.0],
                   'etach': [0.5, 0.5, 0.9, 0.7],
                   'etacl': [0.25, 0.25, 0.05, 0.1],
                   'lambda_hs': [0.0, 0.99, 0.0, 0.99],
                   'lambda_ls': [0.0, 0.8, 0.0, 0.4],
                   'seeds': [0, 0, 0, 0],
                   'ms': [0, 0, 1, 1],
                   'elig': [NO_TRACE, A_TRACE, NO_TRACE, A_TRACE]}
        
        fault = 'damp_elevator_and_saturate_elevator'  # Available: 'none', 'shift_cg', 'damp_elevator'

        env_config = {'state_dim'     : 4,
                      'action_dim'    : 3,
                      'trim_input'    : trim_input,
                      'trim_state'    : trim_state,
                      'dt'            : dt,
                      't_end'         : t_end,
                      'total_steps'   : total_steps,
                      'fault_time'    : 60,
                      'fault_scenario': fault,
                      'reference'     : {'tracked_state' : ['phi', 'theta', 'psi'],
                                         'signal'        : [0*theta_ref, theta_ref, 0*theta_ref]}}
        
        env = Ce500NonLinear(env_config)
        DIRECTORY = f'exps/'
        N_runs = len(configs['etaah'])
        
        print(f'Performing MC test on {N_runs} configurations for fault: {fault}')
        print(f'saving to {DIRECTORY}\n\n')

        MC_test_hparam(configs, DIRECTORY, env, N_runs, repetitions=100, save=1, show=0, transparency=0.1)

        toc = timeit.default_timer()    
        print(f'\n\nElapsed time: {toc-tic:.2f} s')
    else:
        # Single run
        tic = timeit.default_timer()

        # set seed for reproducibility
        seed = 8

        env_config = {'state_dim'     : 4,
                      'action_dim'    : 3,
                      'trim_input'    : trim_input,
                      'trim_state'    : trim_state,
                      'dt'            : dt,
                      't_end'         : t_end,
                      'total_steps'   : total_steps,
                      'fault_time'    : 60,
                      'fault_scenario': 'none',
                      'reference'     : {'tracked_state' : ['phi', 'theta', 'psi'],
                                          'signal'        : [0*theta_ref, theta_ref, 0*theta_ref]}}
        env = Ce500NonLinear(env_config)

        # IDHP configuration
        n, m = 3, 1
        mdp_s_dim   = env_config['state_dim'] # number of states for the MDP, should be 4: [alpha, theta, q, theta_error]
        idhp_config = {'gamma'        : 0.6,
                       'multistep'    : 0,
                       'lr_decay'     : 0.998,
                       'lambda_h'     : 0.95,
                       'lambda_l'     : 0.95,
                       'kappa'        : [1, 2, 1],
                       'cooldown_time': 2.0,
                       'sigma'        : 0.1,
                       'warmup_time'  : 4.0,            # time to maintain high eta
                       'error_thresh' : 1,
                       'tau'          : 0.02,           # soft update parameter
                       'in_dims'      : mdp_s_dim,      # state + error as network input
                       'actor_config' : {'layers':{10: 'tanh', m: 'tanh'},
                                         'eta_h' : 35.0,
                                         'eta_l' : 5.0,
                                         'elig'  : A_TRACE},
                       'critic_config': {'layers':{10: 'tanh', n: 'linear'},
                                         'eta_h' : 1.4,
                                         'eta_l' : 0.7,
                                         'elig'  : 1233},
                       'rls_config'   : {'state_dim'  : n,
                                          'action_dim': m,
                                          'rls_gamma' : 1,
                                          'rls_cov'   : 10**6}}

        # Set seed for numpy and tensorflow
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

        # Initialize IDHP and begin run
        print(f'Running IDHP with seed {seed}...')
        print(f'using idhp config:\n{idhp_config}\n')
        idhp = IDHPnonlin(env, idhp_config, seed=seed, verbose=1)
        idhp.train()
        
        toc = timeit.default_timer()
        print(f'Simulation elapsed time: {toc-tic:.2f} s')

        DIRECTORY = 'exps/nlin/rls_logging_test/'

        # Begin plots
        plot_idhp_nonlin(idhp, DIRECTORY, save=0, show=1)

        toc = timeit.default_timer()
        print(f'\nTotal elapsed time: {toc-tic:.2f} s')
