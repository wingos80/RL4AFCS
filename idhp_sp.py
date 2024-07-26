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
from objects import IDHPsp
from envs.linear.env import FlyingVShortPeriod, Ce500ShortPeriod
from functions import *

NO_TRACE = None
A_TRACE  = 'accumulating'
R_TRACE  = 'replacing'


if __name__ == '__main__':

    tic        = timeit.default_timer()
    t_end, dt  = 60, 0.02      # in seconds
    T          = 10            # in seconds
    A          = np.deg2rad(5) # in degrees
    ref_lambda = lambda t: A*np.sin(2*np.pi*t/T)
    env_config = {'state_dim' : 2,
                  'action_dim': 1,
                  'x0'        : np.zeros((2, 1)),
                  'dt'        : dt,
                  't_end'     : t_end,
                  'fault_time': 20,
                #   'fault_scenario': 'shift_cg',
                  }


    MC  = 0 # 1 for Monte Carlo simulations, 0 for single run
    tic = timeit.default_timer()
    if MC:
        master_folder = 'citation_step_100runs'
        faults = [None]
        n_faults = len(faults)
        for i in range(n_faults):

            # monte carlo settings
            preselect = 1   # 1 to use preselected hparams, 0 to pick randomly
            n_configs = 1   # number of configurations to run
            seeds     = 10  # number of seeds per configuration 
            save      = 0   # 1 to save plots and data, 0 otherwise
            alpha     = 0.03 # transparanecy of the monte carlo lines
            
            # setting mc env configs
            t_end                        = 60
            fault                        = faults[i]
            ref                          = ref_lambda(np.linspace(0, t_end, int(t_end/dt)))
            env_config['t_end']          = t_end
            env_config['fault_scenario'] = fault
            env_config['fault_time']     = 20
            env_config['reference']      = {'tracked_state': ['alpha'],
                                            'signal'    : [ref]}
            
            env                          = Ce500ShortPeriod(env_config)

            elig_as              = [NO_TRACE, A_TRACE, NO_TRACE, R_TRACE]
            multisteps           = [0,0,2,2]
            kappas               = [800,800,800,800]
            etaahs, etaals       = [2.0,2.0,2.0,2.0], [0.02,0.02,0.02,0.02]
            etachs, etacls       = [0.2,0.2,0.2,0.2], [0.0,0.0,0.0,0.0]
            lambda_hs, lambda_ls = [0.6,0.6,0.6,0.6], [0.2,0.2,0.2,0.2]
            print(f'{COLOR.BLUE}All kappas to be tested:{COLOR.END}\n{kappas}\n')


            n_hparams = len(kappas) # run for n_hparams unique hparam configurations
            for j in range(n_hparams):
                toc = timeit.default_timer()
                print(f'\n\n***********************************')
                print(f'Running hparam config {j+1}/{n_hparams}...')
                print(f'Elapsed time: {toc-tic:.2f} s')
                print(f'***********************************\n\n')

                if preselect:
                    # setting mc idhp configs
                    elig_a             = elig_as[j]
                    multistep          = multisteps[j]
                    kappa              = kappas[j]
                    etaah, etaal       = etaahs[j], etaals[j]
                    etach, etacl       = etachs[j], etacls[j]
                    lambda_h, lambda_l = lambda_hs[j], lambda_ls[j]

                    configs              = pick_discrete_hparams(n_configs)
                    configs['elig_a']    = [elig_a]*n_configs
                    configs['multistep'] = [multistep]*n_configs
                    configs['kappas']    = [kappa]*n_configs
                    configs['lr_a_hs']   = [etaah]*n_configs
                    configs['lr_a_ls']   = [etaal]*n_configs
                    configs['lr_c_hs']   = [etach]*n_configs
                    configs['lr_c_ls']   = [etacl]*n_configs
                    configs['lambda_hs'] = [lambda_h]*n_configs
                    configs['lambda_ls'] = [lambda_l]*n_configs
                else:
                    # setting mc idhp configsa
                    configs               = pick_continuous_hparams(1,
                                                                    lambda_hs= [0.0, 0.0],
                                                                    lambda_ls= [0.0, 0.0],
                                                                    lr_a_hs  = [2.5, 4.7],
                                                                    lr_a_ls  = [0.04, 0.06],
                                                                    lr_c_hs  = [0.45, 0.55],
                                                                    lr_c_ls  = [0.0, 0.0],
                                                                    kappas   = [1150, 1550],
                                                                    )
                    
                    kappa    = configs['kappas'][0]
                    etaah    = configs['lr_a_hs'][0]
                    etaal    = configs['lr_a_ls'][0]
                    etach    = configs['lr_c_hs'][0]
                    etacl    = configs['lr_c_ls'][0]
                    lambda_h = configs['lambda_hs'][0]
                    lambda_l = configs['lambda_ls'][0]
                print(configs)

                dirr = f'exps/{master_folder}/{fault}/k{kappa}_etaah{etaah}_etaal{etaal}_etach{etach}_etacl{etacl}_lh{lambda_h}_ll{lambda_l}/'

                print('\n----------------------------------------------------------')
                print(f"Running Monte Carlo simulations...")
                print(f'    {n_configs} configurations with {seeds} seeds each.')
                print(f'    Saving to dir: {dirr}')
                print('----------------------------------------------------------\n')
                
                MC_run(n_configs, configs, env, env_config, seeds, alpha, dirr, save)
    else:
        # config for citation short period
        idhp_config = {'gamma'      : 0.6,
                    'multistep'     : 2,    
                    'gamma_rls'    : 1.0,
                    'lambda_h'     : 0.576,
                    'lambda_l'     : 0.296,
                    'kappa'        : 1140,
                    'cooldown_time': 2.0,
                    'sigma'        : 0.1,
                    'warmup_time'  : 3.0,   # time to maintain high eta
                    'error_thresh' : 1,
                    'tau'          : 0.01, # soft update parameter
                    'in_dims'      : 1,    # state + error as network input
                    'actor_config' : {'layers':{4: 'tanh', env_config['action_dim']: 'tanh'},
                                        'eta_h' : 3.55,
                                        'eta_l' : 0.054,
                                        'elig'  : None},
                    'critic_config': {'layers':{4: 'tanh', env_config['state_dim']: 'linear'},
                                        'eta_h' : 0.338,
                                        'eta_l' : 0.00,
                                        'elig'  : None},
                    'rls_config'   : {'state_dim' : env_config['state_dim'],
                                        'action_dim': env_config['action_dim'],
                                        'rls_gamma' : 1,
                                        'rls_cov'   : 10**6}}
        ref                          = ref_lambda(np.linspace(0, t_end, int(t_end/dt)))
        env_config['reference']      = {'tracked_state': ['alpha'],
                                        'signal'       : [ref]}
        
        env_config['fault_scenario'] = None
        env   = Ce500ShortPeriod(env_config)
        seed = 4

        # set seed for numpy and tensorflow
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

        print(f'Running IDHP with seed {seed}...')

        print(f'using idhp config:\n{idhp_config}')
        idhp = IDHPsp(env, idhp_config, seed=seed, verbose=0)
        idhp.train()
        

        # begin plots
        toc = timeit.default_timer()
        print(f'Elapsed time: {toc-tic:.2f} s')
        single_run(idhp, dt, t_end, save=0)


    toc = timeit.default_timer()
    print(f'\n\nElapsed time: {toc-tic:.2f} s')
