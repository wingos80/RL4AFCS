import os, pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing.pool import Pool
import timeit
from objects import IDHPsp, IDHPnonlin
from utils import *
from tqdm import tqdm
import tensorflow as tf

plt.rcParams["font.family"] = "Arial"


def MC_run_seed(seed, env, idhp_config):
    print(f'PID {os.getpid()}: Running IDHP with seed {seed}...')
    idhp = IDHPsp(env, idhp_config, seed=seed, verbose=0)
    idhp.train()

    output = {}
    output['x_array']        = idhp.x_hist
    output['a_array']        = np.squeeze(idhp.a_hist)
    output['c_array']        = idhp.c_hist
    output['wa_array']       = np.linalg.norm(idhp.a_weights_hist1, axis=1) + np.linalg.norm(idhp.a_weights_hist2, axis=1)
    output['wc_array']       = np.linalg.norm(idhp.c_weights_hist1, axis=1) + np.linalg.norm(idhp.c_weights_hist2, axis=1)
    output['p_array']        = np.linalg.norm(idhp.params_hist, axis=1)
    output['cov_array']      = np.linalg.norm(idhp.cov_hist, axis=1)
    output['eps_array']      = idhp.eps_norm_hist
    episode_return           = np.sum(idhp.c_hist)/idhp.env.kappa
    output['sum_c_array']    = episode_return
    output['a_grad']         = idhp.a_grad_hist
    output['c_grad']         = idhp.c_grad_hist
    convergence_time         = get_convergence_time(idhp.c_hist, idhp.env.kappa, env.dt)
    output['converged_time'] = convergence_time
    output['ref_hist']       = idhp.ref_hist
    return output
    
def MC_run(n_configs, configs, env, env_config, seeds, alpha, dirr, save=False):
    """
    """
    tic       = timeit.default_timer() # measure wallclock time
    dt, t_end = env_config['dt'], env_config['t_end']

    for config in range(n_configs):
        
        directory = dirr
        tomc      = timeit.default_timer()

        print(f"config: {config}")
        # citation settings
        lamb_h        = 0.34  if configs['lambda_hs']      is None else configs['lambda_hs'][config]
        lamb_l        = 0.0   if configs['lambda_ls']      is None else configs['lambda_ls'][config]
        kappa         = 1200  if configs['kappas']         is None else configs['kappas'][config]
        cooldown_time = 2.0   if configs['cooldown_times'] is None else configs['cooldown_times'][config]
        sigma         = 0.1   if configs['sigmas']         is None else configs['sigmas'][config]
        warmup_time   = 3.0   if configs['warmup_times']   is None else configs['warmup_times'][config]
        elig_a        = None  if configs['elig_a']         is None else configs['elig_a'][config]
        lr_a_h        = 3.0   if configs['lr_a_hs']        is None else configs['lr_a_hs'][config]
        lr_a_l        = 0.05  if configs['lr_a_ls']        is None else configs['lr_a_ls'][config]
        lr_c_h        = 0.5   if configs['lr_c_hs']        is None else configs['lr_c_hs'][config]
        lr_c_l        = 0.00  if configs['lr_c_ls']        is None else configs['lr_c_ls'][config]
        multistep     = 0     if configs['multistep']      is None else configs['multistep'][config]
        
        factor        = 1
        idhp_config = {'multistep'    : multistep,
                       'gamma'        : 0.6,
                       'gamma_rls'    : 1.0,
                       'lambda_h'     : lamb_h,
                       'lambda_l'     : lamb_l,
                       'kappa'        : kappa,
                       'cooldown_time': cooldown_time,
                       'sigma'        : sigma,
                       'warmup_time'  : warmup_time,   # time to maintain high eta
                       'error_thresh' : 1,
                       'tau'          : 0.01,
                       'in_dims'      : 1,    # state + error as network input
                       'actor_config' : {'layers':{4: 'tanh', env_config['action_dim']: 'tanh'},
                                         'eta_h' : lr_a_h/factor,
                                         'eta_l' : lr_a_l/factor,
                                         'elig'  : elig_a},
                       'critic_config': {'layers':{4: 'tanh', env_config['state_dim']: 'linear'},
                                         'eta_h' : lr_c_h/factor,
                                         'eta_l' : lr_c_l/factor,
                                         'elig'  : None},
                       'rls_config'   : {'state_dim' : env_config['state_dim'],
                                         'action_dim': env_config['action_dim'],
                                         'rls_gamma' : 1,
                                         'rls_cov'   : 10**6}}
        print(f'{idhp_config}\n\n')

        arrays   = {'x_array'       : np.zeros((seeds, int(t_end/dt), 2)),
                    'a_array'       : np.zeros((seeds, int(t_end/dt))),
                    'wa_array'      : np.zeros((seeds, int(t_end/dt))),
                    'wc_array'      : np.zeros((seeds, int(t_end/dt))),
                    'c_array'       : np.zeros((seeds, int(t_end/dt))),
                    'p_array'       : np.zeros((seeds, int(t_end/dt))),
                    'eps_array'     : np.zeros((seeds, int(t_end/dt))),
                    'sum_c_array'   : np.zeros((seeds)),
                    'cov_array'     : np.zeros((seeds, int(t_end/dt))),
                    'converged_time': np.zeros((seeds)),
                    'a_grad'        : np.zeros((seeds, int(t_end/dt))),
                    'c_grad'        : np.zeros((seeds, int(t_end/dt)))}
        diverged             = 0
        unsteady_convergence = 0
        
        # running 1 python process for each IDHP run
        results    = []
        batch_size = 10
        for i in range(0, seeds, batch_size):
            print(f'running seeds from {i} to {i+batch_size} for config: {config}')
            print(f'using batch size of {batch_size}')
            with Pool() as pool:
                results += [pool.apply_async(MC_run_seed, args=(seed, env, idhp_config)) for seed in range(i, i+batch_size)]
                pool.close()
                pool.join()
        
        for i in range(seeds):            
            output = results[i].get()

            arrays['x_array'][i]        = output['x_array']
            arrays['a_array'][i]        = output['a_array']
            arrays['c_array'][i]        = output['c_array']
            arrays['wa_array'][i]       = output['wa_array']
            arrays['wc_array'][i]       = output['wc_array']
            arrays['p_array'][i]        = output['p_array']
            arrays['cov_array'][i]      = output['cov_array']
            arrays['eps_array'][i]      = output['eps_array']
            episode_return              = output['sum_c_array']
            arrays['sum_c_array'][i]    = episode_return
            arrays['a_grad'][i]         = output['a_grad']
            arrays['c_grad'][i]         = output['c_grad']
            convergence_time            = output['converged_time']
            arrays['converged_time'][i] = convergence_time

            toc     = timeit.default_timer()

            # check if the agent diverged
            if np.isnan(arrays['x_array'][i]).any():
                diverged += 1

            if convergence_time > 30:
                unsteady_convergence += 1
        ref_hist = output['ref_hist']
        # calculate PSD for state and reference
        aoa_PSD        = arrays['x_array'][:,:,0]
        aoa_PSD, omega = get_PSD(t_end, dt, aoa_PSD)
        ref_PSD, _     = get_PSD(t_end, dt, ref_hist)
        PSD_err        = np.sum((aoa_PSD - ref_PSD)**2,axis=-1) # sum of squared errors, 

        # begin plots and logging
        colors      = ['C0','C1','C2','C0','C1','C2']

        # make all the indicators for the run
        # 1. remove diverged runs from the samples
        PSD_err                  = np.delete(PSD_err, np.where(np.isnan(PSD_err)))
        arrays['sum_c_array']    = np.delete(arrays['sum_c_array'], np.where(np.isnan(arrays['sum_c_array'])))
        arrays['converged_time'] = np.delete(arrays['converged_time'], np.where(np.isnan(arrays['sum_c_array'])))

        # 2. calculate the averages
        avg_PSD_err = np.around(np.average(PSD_err), decimals=4)
        avg_c       = np.around(np.average(arrays['sum_c_array']), decimals=4)
        avg_t       = np.around(np.average(arrays['converged_time']), decimals=4)
        toc         = timeit.default_timer()

        # print all the indicators for the run
        print("\n\n---------------------------------------------")
        print(f"Finished config {config+1}/{n_configs}:")
        print(f"    fault: {env_config['fault_scenario']}")
        print(f"    diverged/total: {diverged}/{seeds}")
        print(f"    convergence beyond 30s: {unsteady_convergence}/{seeds}")
        print(f"    avg PSD error: {avg_PSD_err}")
        print(f"    avg c: {avg_c}")
        print(f"    avg t: {avg_t}")
        print(f"Wall elapsed time: {toc-tic:.2f} s")
        print("---------------------------------------------\n\n")

        traced    = False if elig_a is None else True
        multistep = False if multistep == 0 else True
        config_print = 0
        if multistep:
            config_print += 6
        if traced:
            if elig_a == 'replacing':
                config_print += 1
            elif elig_a == 'accumulating':
                config_print += 2
            
        err_string = f'{avg_PSD_err}'
        return_str = f'{avg_c}'
        t_string   = f'{avg_t}'
        directory  = f'{directory}config-{config_print}_d{diverged}_p{err_string}_c{return_str}_t{t_string}_uc{unsteady_convergence}/'
        # directory  = f'{directory}config-{config}_d{diverged}_p{err_string}_c{return_str}_t{t_string}_uc{unsteady_convergence}/'

        arrays['aoa_PSD']  = aoa_PSD
        arrays['omega']    = omega
        arrays['kappa']    = kappa
        arrays['ref_hist'] = ref_hist

        metrics = {'avg_PSD_err': avg_PSD_err,
                   'avg_c'      : avg_c,
                   'avg_t'      : avg_t,
                   'diverged'   : diverged,
                   'unsteady_convergence': unsteady_convergence}
        
        # hparams_saving(directory, arrays, idhp, idhp_config, dt, t_end, alpha, metrics, save, colors)

        MC_pickling(directory, arrays, idhp_config, metrics, save)        
        MC_plotting(directory, arrays, dt, t_end, alpha, save, colors)

def MC_plotting(directory, arrays, dt, t_end, alpha, save=False, colors=None):
    time = np.linspace(0, t_end, int(t_end/dt))

    omega     = arrays['omega']
    aoa_PSD   = arrays['aoa_PSD']
    wa_array  = arrays['wa_array']
    wc_array  = arrays['wc_array']
    eps_array = arrays['eps_array']
    x_array   = arrays['x_array']
    a_array   = arrays['a_array']
    c_array   = arrays['c_array']
    p_array   = arrays['p_array']
    cov_array = arrays['cov_array']
    a_grad    = arrays['a_grad']
    c_grad    = arrays['c_grad']
    ref_hist  = arrays['ref_hist']
    kappa     = arrays['kappa']
    
    # plot weights
    y_wa   = {r'$w_1$': [wa_array.T, alpha]}
    y_wc   = {r'$w_2$': [wc_array.T, alpha]}
    make_plots(time, [y_wa, y_wc], f'{directory}Weight norms evolution', r'$t$ [s]', [r'||Actor||', r'||Critic||'], save=save, colors=colors)
    # put window on top left
    place_tl()

    # plot history of epsilon norms
    y_eps = {r'$||\epsilon||$': [eps_array.T, alpha]}
    make_plots(time, [y_eps], f'{directory}RLS model error norm', r'$t$ [s]', [r'$||\epsilon||$'], save=save, colors=colors)    
    # put window on bot left
    place_bl()

    aoas = np.rad2deg(x_array[:,:,0].T)
    qs   = np.rad2deg(x_array[:,:,1].T)
    elevator = 20*a_array.T
    
    # plot alpha and q along with the refernce in alpha
    y_alpha = {r'$\delta \alpha$'      : [aoas, alpha],
                r'$\delta \alpha_{ref}$': [np.rad2deg(ref_hist), 1]}
    y_q     = {r'$\delta q$'           : [qs, alpha]}
    y_de    = {r'$\delta d_e$'         : [elevator, alpha]}
    # print(np.rad2deg(x_array[:,:,0].T).shape, np.rad2deg(idhp.ref_hist[:]).shape, np.rad2deg(x_array[:,:,1]).shape, (20*a_array.T).shape)
    figx, axx = make_plots(time, [y_alpha, y_q, y_de], f'{directory}States and action', r'$t$ [s]', [r'$\delta \alpha$ [deg]', r'$\delta q$ [deg/s]', r'$\delta d_e$ [deg]'], save=False, colors=colors, legend=False)
    axx[0].lines[-1].set_linestyle('-.')
    axx[0].lines[-1].set_linewidth(1)
    handl, labl = axx[0].get_legend_handles_labels()
    axx[0].legend(handles=handl[-2:], labels=labl[-2:], loc='upper right')
    if save:
        plt.savefig(f'{directory}States_and_action.pdf')
    # put window on top right
    place_tr()

    # plot history of rewards
    y_c = {r'$-c$': [-c_array.T, alpha]}
    make_plots(time, [y_c], f'{directory}NEGATIVE Reward history', r'$t$ [s]', [r'$-c$ [-]'], save=save, colors=colors, log=[0])
    # put window on bot right
    place_br()

    # plot history of aoa error
    aoa_error = np.rad2deg(np.sqrt(-2*(c_array/kappa)))
    y_c = {r'$\alpha error$': [aoa_error.T, alpha]}
    figx, axx = make_plots(time, [y_c], f'{directory}Alpha error', r'$t$ [s]', [r'$\alpha$ [deg]'], colors=colors, log=[])
    axx[0].axhline(y=0.5, color='grey', linestyle='--')
    if save:
        plt.savefig(f'{directory}Alpha error.pdf')
    # put window on bot right
    place_br()

    # plot history of rls params
    y_params = {r'$\theta$': [p_array.T, alpha]}
    y_covs   = {r'$\Sigma$': [cov_array.T, alpha]}
    make_plots(time, [y_params, y_covs], f'{directory}RLS parameter norms', r'$t$ [s]', [r'||$\theta$||', r'||Cov||'], save=save, colors=colors, log =[1])
    # put window on top right right
    place_trr()

    # plot history of a_grad
    y_a_grad = {r'$||\nabla_a||$': [a_grad.T, alpha]}
    make_plots(time, [y_a_grad], f'{directory}Actor gradient norms', r'$t$ [s]', [r'||$\nabla_a$||'], save=save, colors=colors, log=[0])
    place_brr()

    # plot history of c_grad
    y_c_grad = {r'$||\nabla_c||$': [c_grad.T, alpha]}
    make_plots(time, [y_c_grad], f'{directory}Critic gradient norms', r'$t$ [s]', [r'||$\nabla_c$||'], save=save, colors=colors, log=[0])
    place_brr()

    fig   = plt.figure()
    title = f'{directory}PSD alpha responses'
    fig.canvas.manager.set_window_title(title)
    plt.plot(omega, aoa_PSD.T,color='C0',alpha=alpha)
    plt.xscale('log'); plt.grid()
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Normalized magnitude [-]')
    # put window on bot right right
    place_brr()

    if save:
        plt.savefig(f'{title.replace(" ", "_")}.pdf')
    else:
        plt.show()
    
    plt.close('all')

def MC_pickling(directory, arrays, idhp_config, metrics, save):
    if save:
        print(f"saving to directory: {directory}\n\n")
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created directory: {directory}")

        # pickle the metrics
        with open(f'{directory}metrics.pickle', 'wb') as f:
            pickle.dump(metrics, f)

        with open(f'{directory}metrics.txt', 'w') as f:
            print(metrics, file=f)

        # pickle the idhp configuration
        with open(f'{directory}idhp_config.pickle', 'wb') as f:
            pickle.dump(idhp_config, f)

        with open(f'{directory}idhp_config.txt', 'w') as f:
            print(idhp_config, file=f)

        # save all the arrays in a dictionary and pickle it
        arrays2 = {'x_array'         : np.array(arrays['x_array']),
                    'a_array'        : np.array(arrays['a_array']),
                    'c_array'        : np.array(arrays['c_array']),
                    'eps_array'      : np.array(arrays['eps_array']),
                    'sum_c_array'    : np.array(arrays['sum_c_array']),
                    'aoa_PSD'        : np.array(arrays['aoa_PSD']),
                    'omega'          : np.array(arrays['omega']),
                    'converged_times': np.array(arrays['converged_time']),
                    'critic_weights': arrays['wc_array'],
                    'actor_weights' : arrays['wa_array']}

        with open(f'{directory}arrays2.pickle', 'wb') as f:
            pickle.dump(arrays2, f)

def hparams_saving(directory, arrays, idhp, idhp_config, dt, t_end, alpha, metrics, save=False, colors=None):

    # save the idhp config
    if save:
        print(f"saving to directory: {directory}\n\n")
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created directory: {directory}")

        # pickle the idhp configuration
        with open(f'{directory}idhp_config.pickle', 'wb') as f:
            pickle.dump(idhp_config, f)

        with open(f'{directory}idhp_config.txt', 'w') as f:
            print(idhp_config, file=f)

        with open(f'{directory}metrics.txt', 'w') as f:
            print(metrics, file=f)

        with open(f'{directory}metrics.pickle', 'wb') as f:
            pickle.dump(metrics, f)

        
    # plotting only the hiistory of rewards and psd
    time = np.linspace(0, t_end, int(t_end/dt))

    omega     = arrays['omega']
    aoa_PSD   = arrays['aoa_PSD']
    wa_array  = arrays['wa_array']
    wc_array  = arrays['wc_array']
    eps_array = arrays['eps_array']
    x_array   = arrays['x_array']
    a_array   = arrays['a_array']
    c_array   = arrays['c_array']
    p_array   = arrays['p_array']
    cov_array = arrays['cov_array']
    

    # plot history of rewards
    y_c = {r'$-c$': [-c_array.T, alpha]}
    title = f'{directory}NEGATIVE Reward history'
    make_plots(time, [y_c], title, r'$t$ [s]', [r'$-c$ [rad]'], save=False, colors=colors, log=[0])
    plt.ylim(5*10**(-10), 10**(0))
    if save:
        plt.savefig(f'{title.replace(" ", "_")}.pdf')
    # put window on bot right
    place_br()

    aoas = np.rad2deg(x_array[:,:,0].T)
    qs   = np.rad2deg(x_array[:,:,1].T)
    elevator = 20*a_array.T
    # plot alpha and q along with the refernce in alpha
    y_alpha = {r'$\delta \alpha$'      : [aoas, alpha],
                r'$\delta \alpha_{ref}$': [np.rad2deg(idhp.ref_hist[:]), 1]}
    y_q     = {r'$\delta q$'           : [qs, alpha]}
    y_de    = {r'$\delta d_e$'         : [elevator, alpha]}
    title = f'{directory}States and action'
    # print(np.rad2deg(x_array[:,:,0].T).shape, np.rad2deg(idhp.ref_hist[:]).shape, np.rad2deg(x_array[:,:,1]).shape, (20*a_array.T).shape)
    make_plots(time, [y_alpha, y_q, y_de], title, r'$t$ [s]', [r'$\delta \alpha$ [deg]', r'$\delta q$ [deg]', r'$\delta d_e$ [deg]'], save=save, colors=colors)
    # put window on top right
    place_tr()

    fig   = plt.figure()
    title = f'{directory}PSD alpha responses'
    fig.canvas.manager.set_window_title(title)
    plt.plot(omega, aoa_PSD.T,color='C0',alpha=alpha)
    plt.xscale('log'); plt.grid()
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Normalized magnitude')
    # put window on bot right right
    place_brr()

    if save:
        plt.savefig(f'{title.replace(" ", "_")}.pdf')
    else:
        plt.show()
    
    plt.close('all')

def single_run(idhp, dt, t_end, save=False):
    t1, t2 = 0, t_end
    
    y_alpha = {r'$\delta \alpha$': [np.rad2deg(idhp.x_hist[:,0]), 1], 
               r'$\delta \alpha_{ref}$': [np.rad2deg(idhp.ref_hist[:]), 1]}
    # calculate PSD for state and reference
    aoa_PSD        = idhp.x_hist[:,0]
    aoa_PSD, omega = get_PSD(t_end, dt, aoa_PSD)
    ref_PSD, _     = get_PSD(t_end, dt, idhp.ref_hist[:])
    PSD_err        = np.sum((aoa_PSD - ref_PSD)**2,axis=-1) # sum of squared errors, 

    diverged = np.isnan(idhp.x_hist).any()
    cc       = np.sum(idhp.c_hist)/idhp.env.kappa

    actual_A = np.eye(2)+dt*idhp.env.A
    actual_B = dt*idhp.env.B


    # print all the indicators for the run
    print("\n\n---------------------------------------------")
    print(f"Finished experiment:")
    print(f'    actual model    : \n{np.concatenate((actual_A, actual_B), axis=1)}')
    print(f'    identified model: \n{idhp.model.params.T}')
    print(f"    diverged        : {diverged}")
    print(f"    PSD error       : {PSD_err:.6f}")
    print(f"    return          : {cc:.6f}")
    print("---------------------------------------------")


    directory = 'IDHP/exp/casper_hparam/'
    # save the idhp config
    if save:
        print(f"saving to directory: {directory}\n\n")
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created directory: {directory}")
    # find convergence time, converged defined as when error < 0.05 (1% of sine wave amplitude)
    # converged_timestep = np.where(np.rad2deg(idhp.c_hist) < -0.05)[0][-1]
    # converged_time     = converged_timestep*dt

    time = np.linspace(0, t_end, int(t_end/dt))
    colors=['C0','C1','C2','C0','C1','C2']
    # plotting network weight evolutions
    y_aw1     = {r'$w_1$': [idhp.a_weights_hist1, 1]}
    y_aw2     = {r'$w_2$': [idhp.a_weights_hist2, 1]}
    y_cw1     = {r'$w_1$': [idhp.c_weights_hist1, 1]}
    y_cw2     = {r'$w_2$': [idhp.c_weights_hist2, 1]}
    fig1, ax1 = make_plots(time, [y_aw1, y_aw2, y_cw1, y_cw2], f'{directory}Weights evolution', r'$t$ [s]', [r'Actor $w_1$', r'Actor $w_2$', r'Critic $w_1$', r'Critic $w_2$'])
    # ax1[0].set_ylim(-80, 80)
    # ax1[1].set_ylim(-80, 80)
    # ax1[2].set_ylim(-80, 80)
    # ax1[3].set_ylim(-80, 80)
    plt.xlim(t1,t2)
    # put window on top left
    # place_tl()
    fig1.set_size_inches(4,3.2)
    plt.subplots_adjust(left=0.2, top=0.96,bottom=0.15, right=0.98)
    if save:
        with open(f'{directory}Weights evolution', 'wb') as f:
            pickle.dump(fig1, f)

    # plot idhp.model.eps_norm_hist
    y_eps = {r'$||\epsilon||$': [idhp.eps_norm_hist, 1]}
    fig2, ax2 = make_plots(time, [y_eps], f'{directory}RLS model error norm', r'$t$ [s]', [r'$||\epsilon||$'], colors=colors, log=[0],save=save)
    # put window on bot left
    plt.xlim(t1,t2)
    place_bl()
    if save:
        with open(f'{directory}RLS model error norm', 'wb') as f:
            pickle.dump(fig2, f)


    # plot alpha and q along with the refernce in alpha
    y_alpha = {r'$\alpha$': [np.rad2deg(idhp.x_hist[:,0]), 1], 
               r'$\alpha_{ref}$': [np.rad2deg(idhp.ref_hist[:]), 1]}
    y_q     = {r'$q$': [np.rad2deg(idhp.x_hist[:,1]), 1]}
    y_de    = {r'$d_e$': [20*idhp.a_hist, 1]}
    fig3, ax3 = make_plots(time, [y_alpha, y_q, y_de], f'{directory}States and action', r'$t$ [s]', [r'$\alpha$ [deg]', r'$q$ [deg]', r'$d_e$ [deg]'], colors=colors,save=save)
    ax3[0].lines[1].set_linewidth(1)
    ax3[0].lines[1].set_linestyle('-.')
    # ax3[0].axvline(x=20, color='k', linestyle='-.', label='Fault', alpha = 0.5)
    ax3[0].legend(loc='upper right')
    ax3[1].axvline(x=20, color='k', linestyle='-.', label='Fault', alpha = 0.5)
    ax3[2].axvline(x=20, color='k', linestyle='-.', label='Fault', alpha = 0.5)
    
    # put window on top right
    plt.xlim(t1,t2)
    place_tr()
    if save:
        with open(f'{directory}States and action', 'wb') as f:
            pickle.dump(fig3, f)

    # plot history of rewards
    y_c = {r'$-c$': [-idhp.c_hist, 1]}
    fig4, ax4 = make_plots(time, [y_c], f'{directory}NEGATIVE Reward history', r'$t$ [s]', [r'$-c$ [-]'], colors=colors, log=[0],save=save)
    # put window on bot right
    plt.xlim(t1,t2)
    place_br()
    if save:
        with open(f'{directory}NEGATIVE Reward history', 'wb') as f:
            pickle.dump(fig4, f)

    # plot history of aoa error
    aoa_error = np.rad2deg(np.sqrt(-2*(idhp.c_hist/idhp.env.kappa)))
    y_c = {r'$\alpha error$': [aoa_error, 1]}
    fig4, ax4 = make_plots(time, [y_c], f'{directory}Alpha error', r'$t$ [s]', [r'$\alpha$ [deg]'], colors=colors, log=[0],save=save)
    # plot horizontal dashed line at 0.5
    ax4[0].axhline(y=0.5, color='r', linestyle='--')
    # put window on bot right
    plt.xlim(t1,t2)
    place_br()
    if save:
        with open(f'{directory}Alpha error', 'wb') as f:
            pickle.dump(fig4, f)


    # plot history of rls params & covariances
    y_params = {r'$\theta$': [idhp.params_hist, 1]}
    y_covs   = {r'$\Sigma$': [idhp.cov_hist, 1]}
    fig5, ax5 = make_plots(time, [y_params, y_covs], f'{directory}RLS parameters', r'$t$ [s]', [r'$\theta$', r'$\sigma$'],save=save)
    # put window on top right right
    plt.xlim(t1,t2)
    place_trr()
    if save:
        with open(f'{directory}RLS parameters', 'wb') as f:
            pickle.dump(fig5, f)

    fig12 = plt.figure()
    title = f'PSD alpha responses'
    fig12.canvas.manager.set_window_title(title)

    alpha = idhp.x_hist[:,0]
    alpha_fft, omega = get_PSD(t_end, dt, alpha)
    plt.plot(omega, alpha_fft,color='C0',alpha=1)
    plt.xscale('log')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Normalized magnitude')
    # put window on bot right right
    place_trr()
    if save:
        plt.savefig(f'{directory}{title.replace(" ", "_")}.pdf')

        with open(f'{directory}PSD alpha responses', 'wb') as f:
            pickle.dump(fig12, f)
    
    # plot history of a_grad
    y_a_grad = {r'$||\nabla a||$': [idhp.a_grad_hist, 1]}
    fig7, ax7 = make_plots(time, [y_a_grad], f'{directory}Actor gradient norms', r'$t$ [s]', [r'||$\nabla a$||'], colors=colors, log=[0],save=save)
    plt.xlim(t1,t2)
    place_brr()
    if save:
        with open(f'{directory}Actor gradient norms', 'wb') as f:
            pickle.dump(fig7, f)

    # plot history of a_grad_all
    y_a_grad_all = {r'$\nabla a$': [idhp.a_all_grad_hist, 1]}
    fig6, ax6 = make_plots(time, [y_a_grad_all], f'{directory}Actor gradients', r'$t$ [s]', [r'$\nabla a$'],save=save)
    # plt.ylim(-80, 80)
    plt.xlim(t1,t2)
    plt.ylim(-0.16, 0.16)
    plt.subplots_adjust(left=0.22, top=0.96,bottom=0.20, right=0.98)
    place_smol()
    if save:
        with open(f'{directory}Actor gradients', 'wb') as f:
            pickle.dump(fig6, f)

    # plot history of c_grad
    y_c_grad = {r'$||\nabla c||$': [idhp.c_grad_hist, 1]}
    fig9, ax9 = make_plots(time, [y_c_grad], f'{directory}Critic gradient norms', r'$t$ [s]', [r'||$\nabla c$||'], colors=colors, log=[0],save=save)
    plt.xlim(t1,t2)
    place_brr()
    if save:
        with open(f'{directory}Critic gradient norms', 'wb') as f:
            pickle.dump(fig9, f)

    # plot history of c_grad_all
    y_c_grad_all = {r'$\nabla c$': [idhp.c_all_grad_hist, 1]}
    fig8, ax8 = make_plots(time, [y_c_grad_all], f'{directory}Critic gradients', r'$t$ [s]', [r'$\nabla c$'],save=save)
    # plt.ylim(-80, 80)
    plt.xlim(t1,t2)
    plt.ylim(-30, 40)
    plt.subplots_adjust(left=0.18, top=0.96,bottom=0.20, right=0.98)
    place_smol()
    if save:
        with open(f'{directory}Critic gradients', 'wb') as f:
            pickle.dump(fig8, f)

    # plot critic eligibility trace
    y_c_e  = {r'elig_c': [idhp.c_e_hist, 1]}
    fig10, ax10 = make_plots(time, [y_c_e], f'{directory}Critic eligibility trace', r'$t$ [s]', [r'elig_c'],save=save)
    plt.xlim(t1,t2)
    place_bl()
    if save:
        with open(f'{directory}Critic eligibility trace', 'wb') as f:
            pickle.dump(fig10, f)

    # plot actor eligibility trace
    y_a_e = {r'elig_a': [idhp.a_e_hist, 1]}
    fig11, ax11 = make_plots(time, [y_a_e], f'{directory}Actor eligibility trace', r'$t$ [s]', [r'elig_a'],save=save)
    plt.xlim(t1,t2)
    place_br()
    if save:
        with open(f'{directory}Actor eligibility trace', 'wb') as f:
            pickle.dump(fig11, f)

    # plot actor gain over time with alpha response shared time-axis

    fig12,ax12 = plt.subplots(2,1,sharex=True)
    ax12[0].plot(time,np.rad2deg(idhp.x_hist[:,0]),color='C0',label=r'$\alpha$')
    ax12[0].plot(time,np.rad2deg(idhp.ref_hist[:]),color='C1',label=r'$\alpha_{ref}$',linewidth=1,linestyle='-.')
    ax12[0].axvline(x=20, color='k', linestyle='-.', label='Fault', alpha = 0.5)
    ax12[0].set_ylabel(r'$\alpha$ [deg]')
    # ax12[0].set_xlabel(r'$t$ [s]')
    ax12[0].grid()
    ax12[0].legend(loc='upper right')

    errors      = np.deg2rad(np.linspace(-5,5,100))
    deflections = np.zeros((int(t_end/dt),100))
    aw1         = idhp.a_weights_hist1
    aw2         = idhp.a_weights_hist2

    for i, error in enumerate(errors):
        deflection = error*aw1
        deflection = np.tanh(deflection)
        deflection = deflection@aw2.T
        deflection = 20*np.tanh(np.diag(deflection))

        deflections[:, i] = deflection.flatten()

    y=np.linspace(-5,5,101)
    x=np.linspace(0,t_end,len(time)+1)
    Z=deflections.T
    im=ax12[1].pcolormesh(x,y,Z,cmap=plt.cm.RdYlGn)
    im.set_clim(-20,20)
    # fig12.colorbar(im,ax=ax12[1],label=r'$\delta_e$ [deg]', location='bottom')
    ax12[1].set_ylabel(r'$e$ [deg]')
    ax12[1].set_xlabel(r'$t$ [s]')
    fig12=plt.gcf()
    fig12.set_size_inches(6.2,2.7)
    plt.tight_layout()
        
    timess = [0.5, 19.5, 35]
    for tim in timess:
        plot_policy(tim, errors, deflections)

    # plot the reward gradient function estimate
    cw1 = idhp.c_weights_hist1[-1,:].reshape(-1,1)
    cw2 = idhp.c_weights_hist2[-1,:].reshape(4,2)
    gradients = np.zeros((100,2))
    for i,error in enumerate(errors):
        gradients[i,:] = (cw2.T@np.tanh(error*cw1)).flatten()
    plt.figure()
    plt.plot(gradients)
    plt.xlabel('final')

    # plot the reward gradient function estimate
    cw1 = idhp.c_weights_hist1[0,:].reshape(-1,1)
    cw2 = idhp.c_weights_hist2[0,:].reshape(4,2)
    gradients = np.zeros((100,2))
    for i,error in enumerate(errors):
        gradients[i,:] = (cw2.T@np.tanh(error*cw1)).flatten()
    plt.figure()
    plt.plot(gradients)
    plt.xlabel('initial')


    # # plot the epislons from idhp
    # y_eps = {r'$\epsilon_{\alpha}$': [np.rad2deg(idhp.eps_hist[:,0]), 1],
    #          r'$\epsilon_{q}$'     : [np.rad2deg(idhp.eps_hist[:,1]), 1]}
    # fig12, ax12 = make_plots(time, [y_eps], f'{directory}Epsilon history', r'$t$ [s]', [r'$\epsilon$'],save=save)
    plt.show()

def plot_policy(time, errors, deflections):

    
    timestep = int(time*50)
    figg,axx = plt.subplots(figsize=(4,3.2))
    figg.canvas.manager.set_window_title(f'policy_at_{timestep*0.02}_s') 

    axx.plot(np.rad2deg(errors),deflections[timestep,:])

    # set the x-spine (see below for more info on `set_position`)
    axx.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    axx.spines['right'].set_color('none')
    axx.yaxis.tick_left()

    # set the y-spine
    axx.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    axx.spines['top'].set_color('none')
    axx.xaxis.tick_bottom()

    axx.set_ylabel(r'$\quad \delta_e$ [deg]', loc='top', rotation=0)
    axx.set_xlabel(r'$e$ [deg]', loc='right')
    axx.xaxis.set_label_coords(1.1,0.6)
    axx.yaxis.set_label_coords(0.6,1.06)
    axx.plot(1, 0, ">k", transform=axx.get_yaxis_transform(), clip_on=False)
    axx.plot(0, 1, "^k", transform=axx.get_xaxis_transform(), clip_on=False)
    axx.set_ylim(-20,20)
    
def plot_idhp_nonlin(idhp, directory, save=0, show=1):
    """
    Plotting and printing the results of one idhp run on the nonlinear citation model
    """
    colors = ['C0','C1','C2','C0','C1','C2']
    index  = 7
    name   = 'theta'
    unit   = '[deg]'

    times   = idhp.log['t']

    states        = idhp.log['x_full']           # obtain states
    states[:,:3]  = np.rad2deg(states[:,:3])     # convert attitude rates to deg/s
    states[:,4:9] = np.rad2deg(states[:,4:9])    # convert attitudes to deg

    refs_lon = np.rad2deg(idhp.log['yref'])   # obtain and convert references to deg
    error    = np.rad2deg(idhp.log['e'])      # obtain and convert error to deg

    action_commanded = np.rad2deg(idhp.log['a_cmd'][:,0]) # obtain and convert elevator deflections to deg
    action_effective = np.rad2deg(idhp.log['a_eff'][:,0]) # obtain and convert control deflections to deg
    
    # compute normal acceleration during the flight, n_z = V*q
    n_z = states[:,3]*np.deg2rad(states[:,1])/9.80665

    # save the idhp config
    if save:
        print(f"saving to directory: {directory}\n\n")
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created directory: {directory}")
        with open(f'{directory}log.pickle', 'wb') as f:
            pickle.dump(idhp.log, f)

    # plot the learning rate curve
    fig, ax = plt.subplots()
    title = 'eta_a'
    fig.canvas.manager.set_window_title(f'{title}')
    ax.plot(times, idhp.log['eta_a'])
    ax.set_ylabel(r'$\eta_a$')
    ax.set_xlabel(r'$t$ [s]')
    ax.grid()
    ax.set_xlim(0, times[-1])
    place_trr()
    if save:
        fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        with open(f'{directory}{title.replace(" ", "_")}.pickle', 'wb') as f:
            pickle.dump(fig, f)
            

    # plot various longitudinal system states
    fig, ax = plt.subplots(8,1,sharex=True)
    title = name + '_tracking_p1'
    fig.canvas.manager.set_window_title(f'{title}')
    ax[0].plot(times, n_z)
    ax[0].set_ylabel(r'$g$ [-]')
    ax[0].grid()
    ax[0].set_ylim(-11, 11)
    ax[1].plot(times, states[:,1])
    ax[1].set_ylabel(r'$q$ [deg/s]')
    ax[1].grid()
    ax[1].set_ylim(-45, 45)
    ax[2].plot(times, states[:,4])
    ax[2].set_ylabel(r'$\alpha$ [deg]')
    ax[2].grid()
    ax[2].set_ylim(0.8, 19.1)
    ax[3].plot(times, states[:,7])
    ax[3].set_ylabel(r'$\theta$ [deg]')
    ax[3].grid()
    ax[3].set_ylim(-2.7, 23)
    ax[4].plot(times, states[:,3])
    ax[4].set_ylabel(r'$V_{TAS}$ [m/s]')
    ax[4].grid()
    ax[4].set_ylim(74, 99)
    ax[5].plot(times, states[:,-3])
    ax[5].set_ylabel(r'$h$ [m]')
    ax[5].grid()
    ax[5].set_ylim(1940, 2520)
    ax[6].plot(times, action_commanded)
    ax[6].set_ylabel(r'$\delta_{e, cmd}$ [deg]')
    ax[6].grid()
    ax[6].set_ylim(-16, 16)
    ax[7].plot(times, action_effective)
    ax[7].set_ylabel(r'$\delta_{e, eff}$ [deg]')
    ax[7].grid()
    ax[7].set_ylim(-16, 16)
    ax[-1].set_xlabel(r'$t$ [s]')
    ax[-1].set_xlim(0, times[-1])
    place_l()
    fig.set_size_inches(7,8.4)
    fig.subplots_adjust(left=0.11, top=0.96,bottom=0.075, right=0.98)
    if save:
        fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        with open(f'{directory}{title.replace(" ", "_")}.pickle', 'wb') as f:
            pickle.dump(fig, f)

    # plot the longitudinal tracking performance
    fig, ax = plt.subplots(2,1)
    title = name + '_tracking_p2'
    fig.canvas.manager.set_window_title(f'{title}')
    ax[0].plot(times, states[:,index],label=r'$\theta$')
    ax[0].plot(times, refs_lon[:,0]  ,label=r'$\theta_{ref}$'  ,linestyle='-.')
    ax[0].plot(times, error[:,0]     ,label=r'$\theta_{error}$',linestyle='--',linewidth=0.5)
    ax[0].legend()
    ax[0].grid()
    ax[0].set_ylabel(r'$\theta$ [deg]')
    ax[1].plot(times, action_commanded, label=r'$\delta_{e, cmd}$')
    ax[1].grid()
    ax[1].set_ylabel(r'$\delta_{e, cmd}$ [deg]')
    ax[-1].set_xlabel(r'$t$ [s]')
    ax[-1].set_xlim(0, times[-1])
    place_tr()
    fig.set_size_inches(6.15,3.5)
    fig.subplots_adjust(left=0.0190, top=0.96,bottom=0.13, right=0.98)
    if save:
        fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        with open(f'{directory}{title.replace(" ", "_")}.pickle', 'wb') as f:
            pickle.dump(fig, f)

    # plotting network weight evolutions
    y_aw1 = {r'$w_1$': [idhp.log['a_weights1'], 1]}
    y_aw2 = {r'$w_2$': [idhp.log['a_weights2'], 1]}
    y_cw1 = {r'$w_1$': [idhp.log['c_weights1'], 1]}
    y_cw2 = {r'$w_2$': [idhp.log['c_weights2'], 1]}
    title = 'Weights evolution'
    fig1, ax1 = make_plots(times, [y_aw1, y_aw2, y_cw1, y_cw2], f'{directory}{title}', r'$t$ [s]', [r'Actor $w_1$', r'Actor $w_2$', r'Critic $w_1$', r'Critic $w_2$'], save=0)
    ax1[-1].set_xlim(0, times[-1])
    place_br()
    if save:
        fig1.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        with open(f'{directory}{title.replace(" ", "_")}.pickle', 'wb') as f:
            pickle.dump(fig, f)

    # plotting network gradient evolutions
    y_a_grad = {r'$||\nabla a||$': [idhp.log['a_grad'], 1]}
    y_c_grad = {r'$||\nabla c||$': [idhp.log['c_grad'], 1]}
    title = 'Gradient norms'
    fig2, ax2 = make_plots(times, [y_a_grad, y_c_grad], f'{directory}{title}', r'$t$ [s]', [r'||$\nabla a$||', r'||$\nabla c$||'], save=0)
    ax2[-1].set_xlim(0, times[-1])
    place_brr()
    if save:
        fig2.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        with open(f'{directory}{title.replace(" ", "_")}.pickle', 'wb') as f:
            pickle.dump(fig, f)

    # plotting the longitudinal error
    fig, ax = plt.subplots()
    title = 'q_error'
    fig.canvas.manager.set_window_title(f'{title}')
    ax.plot(times, error[:,0])
    ax.set_ylabel(r'$e$ [deg]')
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylim(-4,4)
    ax.set_xlim(0, times[-1])
    ax.grid()
    place_r()
    fig.set_size_inches(6.15,2)
    fig.subplots_adjust(left=0.090, top=0.96,bottom=0.24, right=0.98)
    if save:
        fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        with open(f'{directory}{title.replace(" ", "_")}.pickle', 'wb') as f:
            pickle.dump(fig, f)

    fig, ax = plt.subplots(3,1)
    title = 'RLS_theta_cov_eps'
    fig.canvas.manager.set_window_title(f'{title}')
    ax[0].plot(times, idhp.log['rls_params'])
    ax[0].set_ylabel(r'$\theta$')
    ax[0].grid()
    ax[1].plot(times, idhp.log['rls_cov'])
    ax[1].set_ylabel(r'$\Sigma$')
    ax[1].grid()
    ax[2].plot(times, idhp.log['rls_eps_norm'])
    ax[2].set_ylabel(r'$\epsilon$')
    ax[2].grid()
    ax[-1].set_xlabel(r'$t$ [s]')
    fig.set_size_inches(6.15,4)
    if save:
        fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        with open(f'{directory}{title.replace(" ", "_")}.pickle', 'wb') as f:
            pickle.dump(fig, f)

    RSE_warmup = np.sum(idhp.log['RSE'][:5500,0])
    RSE_flight = np.sum(idhp.log['RSE'][5500:,0])
    theta_PSD, omega = get_PSD(idhp.env.t_end, idhp.env.dt, states[:,7])
    Sm = np.sum(theta_PSD*omega)
    # print all the indicators for the run
    print("\n\n---------------------------------------------")
    print(f"Finished experiment:")
    print(f"    RSE warmup : {RSE_warmup:.3f}")
    print(f"    RSE flight : {RSE_flight:.3f}")
    print(f"    peak gs    : {np.max(np.abs(n_z)):.2f}")
    print(f"    Smoothness : {Sm:.3f}")
    print("---------------------------------------------")
    if show:
        plt.show()

def MC_test_hparam(configs, directory, env, N, repetitions, save=1, show=0, transparency=0.2):
    tic = timeit.default_timer()
    algos  = ['idhp', 'idhprt', 'idhpat', 'midhp', 'midhprt', 'midhpat']

    NO_TRACE = None
    A_TRACE  = 'accumulating'
    R_TRACE  = 'replacing'

    t_end, dt   = env.t_end, env.dt
    total_steps = env.total_steps
    times       = np.linspace(0, t_end, total_steps)
    yref        = np.rad2deg(env.state_reference[1])
    configs  = [{    'etaah': configs['etaah'][i],
                     'etaal': configs['etaal'][i],
                     'etach': configs['etach'][i],
                     'etacl': configs['etacl'][i],
                 'lambda_hs': configs['lambda_hs'][i],
                 'lambda_ls': configs['lambda_ls'][i],
                      'seed': configs['seeds'][i],
                        'ms': configs['ms'][i],
                      'elig': configs['elig'][i]} for i in range(N)]
    
    n, m = 3, 1
    mdp_s_dim = 4
    for i in range(N):
        etaah, etaal = configs[i]['etaah'], configs[i]['etaal']
        etach, etacl = configs[i]['etach'], configs[i]['etacl']
        lh, ll       = configs[i]['lambda_hs'], configs[i]['lambda_ls']
        ms, elig     = configs[i]['ms'], configs[i]['elig']
        if ms == 0:
            if elig == NO_TRACE:
                algo = algos[0]
            elif elig == R_TRACE:
                algo = algos[1]
            elif elig == A_TRACE:
                algo = algos[2]
        elif ms == 1:
            if elig == NO_TRACE:
                algo = algos[3]
            elif elig == R_TRACE:
                algo = algos[4]
            elif elig == A_TRACE:
                algo = algos[5]
        idhp_config = {'gamma'        : 0.6,
                       'multistep'    : ms,
                       'lr_decay'     : 0.998,
                       'lambda_h'     : lh,
                       'lambda_l'     : ll,
                       'kappa'        : [1, 2, 1],
                       'cooldown_time': 2.0,
                       'sigma'        : 0.1,
                       'warmup_time'  : 4,            # time to maintain high eta
                       'error_thresh' : 1,
                       'tau'          : 0.02,           # soft update parameter
                       'in_dims'      : mdp_s_dim,      # state + error as network input
                       'actor_config' : {'layers':{10: 'tanh', m: 'tanh'},
                                           'eta_h' : etaah,
                                           'eta_l' : etaal,
                                           'elig'  : elig},
                       'critic_config': {'layers':{10: 'tanh', n: 'linear'},
                                           'eta_h' : etach,
                                           'eta_l' : etacl,
                                           'elig'  : 1233},
                       'rls_config'   : {'state_dim'  : n,
                                           'action_dim': m,
                                           'rls_gamma' : 1,
                                           'rls_cov'   : 10**6}}
        
        log = {'RSE': np.zeros((repetitions, 2)),
               'e'  : np.zeros((repetitions, total_steps)),
            'theta': np.zeros((repetitions, total_steps)),        # stored in degrees
            'alpha': np.zeros((repetitions, total_steps)),        # stored in degrees
            'q': np.zeros((repetitions, total_steps)),            # stored in degrees
            'V': np.zeros((repetitions, total_steps)),
            'h': np.zeros((repetitions, total_steps)),
            'action_cmd': np.zeros((repetitions, total_steps)),      # stored in degrees
            'action_eff': np.zeros((repetitions, total_steps)),  # stored in degrees
            'n_z': np.zeros((repetitions, total_steps)),
            'wa_norm': np.zeros((repetitions, total_steps)),
            'wc_norm': np.zeros((repetitions, total_steps)),
            'Sm': np.zeros((repetitions, 1)),
            'rls_eps': np.zeros((repetitions, total_steps))}
        
        seeds_used = []
        seed_offset = 0
        print(f'N-{i}, {COLOR.RED}**algo = {algo}**{COLOR.END}')
        print(f'doing {repetitions} repetitions with seed offset: {seed_offset}')
        print(f'using idhp config:\n\n{idhp_config}\n')
        for seed in range(repetitions):
            seed+=seed_offset
            seeds_used.append(seed)

            # set seed for numpy and tensorflow
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf.keras.utils.set_random_seed(seed)

            idhp = IDHPnonlin(env, idhp_config, seed=seed, verbose=0)
            idhp.train()
            seed-=seed_offset
            
            wa_norm = np.linalg.norm(idhp.log['a_weights1'], axis=1)
            wa_norm_normalized = wa_norm/np.max(wa_norm)
            wc_norm = np.linalg.norm(idhp.log['c_weights1'], axis=1)
            wc_norm_normalized = wc_norm/np.max(wc_norm)
            # calculate the smoothness of the in flight pitch angle
            theta_PSD, omega = get_PSD(35, dt, np.rad2deg(idhp.log['x_full'][5500:,7]))
            Sm = np.sum(theta_PSD*omega)

            log['RSE'][seed, 0] = np.sum(idhp.log['RSE'][:5500,0])
            log['RSE'][seed, 1] = np.sum(idhp.log['RSE'][5500:,0])
            log['e'][seed] = np.rad2deg(idhp.log['e']).flatten()
            log['theta'][seed] = np.rad2deg(idhp.log['x_full'][:,7])
            log['alpha'][seed] = np.rad2deg(idhp.log['x_full'][:,4])
            log['q'][seed] = np.rad2deg(idhp.log['x_full'][:,1])
            log['V'][seed] =idhp.log['x_full'][:,3]
            log['h'][seed] = idhp.log['x_full'][:,-3]
            log['action_cmd'][seed] = np.rad2deg(idhp.log['a_cmd'][:,0]).flatten()
            log['action_eff'][seed] = np.rad2deg(idhp.log['a_eff'][:,0]).flatten()
            log['n_z'][seed] = (idhp.log['x_full'][:,3]*idhp.log['x_full'][:,1]/9.80665).flatten()
            log['wa_norm'][seed] = wa_norm_normalized.flatten()
            log['wc_norm'][seed] = wc_norm_normalized.flatten()
            log['Sm'][seed] = Sm
            log['rls_eps'][seed] = idhp.log['rls_eps_norm'].flatten()
            # print all the indicators for the run
            print(f"    seed no.: {seed}|   {COLOR.BLUE}RSE warmup, RSE flight, max nz, Sm : {log['RSE'][seed, 0]:.3f}, {log['RSE'][seed, 1]:.3f}, {np.max(np.abs(log['n_z'][seed])):.2f}, {Sm:.3f}{COLOR.END}")
        
        # dirr = f'{directory}etaah{etaah}_etaal{etaal}_etach{etach}_etacl{etacl}_lh{lh}_ll{ll}/{algo}'
        dirr = f'{directory}/{algo}'
        MC_test_hparam_plot(log, dirr, times, yref, repetitions, idhp_config, seeds_used, tic=tic, transparency=transparency, show=show, save=save)

def MC_test_hparam_plot(log, directory, times, yref, repetitions, idhp_config, seeds_used, tic, transparency=0.2, show=1,save=0):
        RSE_warmup = np.average(log['RSE'][:,0])
        RSE_flight = np.average(log['RSE'][:,1])
        max_abs_nz = np.average(np.max(np.abs(log['n_z']),axis=1))
        avg_Sm     = np.average(log['Sm'])

        directory = f'{directory}_{RSE_warmup:.2f}_{RSE_flight:.2f}/'
        # save the idhp config
        if save:
            print(f"saving to directory: {directory}\n\n")
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"created directory: {directory}")
            with open(f'{directory}log.pickle', 'wb') as f:
                pickle.dump(log, f)
            with open(f'{directory}reps_{repetitions}.txt', 'w') as f:
                f.write(f'{repetitions}')
            # pickle the idhp configuration
            with open(f'{directory}idhp_config.pickle', 'wb') as f:
                pickle.dump(idhp_config, f)
            with open(f'{directory}idhp_config.txt', 'w') as f:
                print(idhp_config, file=f)
            with open(f'{directory}seeds_used.txt', 'w') as f:
                f.write(f'{seeds_used}')

        name = 'theta'
        unit = '[deg]'

        # plot various longitudinal system states
        fig, ax = plt.subplots(8,1,sharex=True)
        title = name + '_tracking_p1'
        fig.canvas.manager.set_window_title(f'{title}')
        ax[0].plot(times, log['n_z'].T, color='C0', alpha=transparency)
        ax[0].set_ylabel(r'$g$ [-]')
        ax[0].grid()
        ax[0].set_ylim(-11, 11)
        ax[1].plot(times, log['q'].T, color='C0', alpha=transparency)
        ax[1].set_ylabel(r'$q$ [deg/s]')
        ax[1].grid()
        ax[1].set_ylim(-45, 45)
        ax[2].plot(times, log['alpha'].T, color='C0', alpha=transparency)
        ax[2].set_ylabel(r'$\alpha$ [deg]')
        ax[2].grid()
        ax[2].set_ylim(0.8, 19.1)
        ax[3].plot(times, log['theta'].T, color='C0', alpha=transparency)
        ax[3].set_ylabel(r'$\theta$ [deg]')
        ax[3].grid()
        ax[3].set_ylim(-5, 23)
        ax[4].plot(times, log['V'].T, color='C0', alpha=transparency)
        ax[4].set_ylabel(r'$V_{TAS}$ [m/s]')
        ax[4].grid()
        ax[4].set_ylim(74, 99)
        ax[5].plot(times, log['h'].T, color='C0', alpha=transparency)
        ax[5].set_ylabel(r'$h$ [m]')
        ax[5].grid()
        ax[5].set_ylim(1940, 2520)
        ax[6].plot(times, log['action_cmd'].T, color='C0', alpha=transparency)
        ax[6].set_ylabel(r'$\delta_{e, cmd}$ [deg]')
        ax[6].grid()
        ax[6].set_ylim(-16, 16)
        ax[7].plot(times, log['action_eff'].T, color='C0', alpha=transparency)
        ax[7].set_ylabel(r'$\delta_{e, eff}$ [deg]')
        ax[7].grid()
        ax[7].set_ylim(-16, 16)
        ax[-1].set_xlabel(r'$t$ [s]')
        ax[-1].set_xlim(times[0], times[-1])
        place_l()
        fig.set_size_inches(7,8.4)
        fig.subplots_adjust(left=0.11, top=0.995,bottom=0.055, right=0.98)
        if save:
            fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')

        # plot the longitudinal tracking performance
        fig, ax = plt.subplots(2,1,sharex=True)
        title = name + '_tracking_p2'
        fig.canvas.manager.set_window_title(f'{title}')
        ax[0].plot(times, log['theta'].T, label=r'$\theta$', linewidth=1, color='C0', alpha=transparency)
        ax[0].plot([-100,-200], [-100, -200], label=r'$\theta$', linewidth=1, color='C0')
        ax[0].plot(times, yref, label=r'$\theta_{ref}$', linewidth=1, linestyle='-.', color='C1')
        handl, labl = ax[0].get_legend_handles_labels()
        ax[0].legend(handles=handl[-2:], labels=labl[-2:], loc='upper right')
        ax[0].scatter([60], [yref[6000]], color='r', marker='x', s=100, linewidth=1.1, zorder=10)
        ax[0].text(61, yref[5660], 'Fault', color='r')
        # ax[0].scatter([60], [yref[500]], color='r', marker='x', s=100, linewidth=1.1, zorder=10)
        # ax[0].text(61, yref[160], 'Fault', color='r')
        ax[0].grid()
        ax[0].set_ylabel(r'$\theta$ [deg]')
        ax[0].set_ylim(-10, 27)
        ax[1].plot(times, log['action_cmd'].T, color='C0', alpha=transparency)
        ax[1].set_ylabel(r"$\delta'_{e}$ [deg]")
        ax[1].grid()
        ax[1].set_ylim(-16, 16)
        ax[-1].set_xlabel(r'$t$ [s]')
        ax[-1].set_xlim(times[0], times[-1])
        place_tr()
        fig.align_ylabels(ax[:])
        # fig.set_size_inches(7.4,3.7)
        fig.set_size_inches(6,3.7)
        fig.subplots_adjust(left=0.100, top=0.96,bottom=0.13, right=0.98)
        # fig.set_size_inches(4.5,3.7)
        # fig.subplots_adjust(left=0.150, top=0.96,bottom=0.13, right=0.98)
        if save:
            fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')

        # plotting network weight evolutions
        y_aw1 = {r'$w_1$': [log['wa_norm'].T, 1]}
        y_cw2 = {r'$w_2$': [log['wc_norm'].T, 1]}
        title = 'Weights evolution'
        fig1, ax1 = make_plots(times, [y_aw1, y_cw2], f'{directory}{title}', r'$t$ [s]', [r'Actor $||w||$', r'Critic $||w||$'], save=0, colors=['C0'])
        ax1[-1].set_xlim(times[0], times[-1])
        place_br()
        if save:
            fig1.savefig(f'{directory}{title.replace(" ", "_")}.pdf')

        # plotting the longitudinal error
        fig, ax = plt.subplots()
        title = 'q_error'
        fig.canvas.manager.set_window_title(f'{title}')
        ax.plot(times, log['e'].T, color='C0', alpha=transparency)
        ax.set_ylabel(r'$e$ [deg]')
        ax.set_xlabel(r'$t$ [s]')
        ax.set_ylim(-4,4)
        ax.set_xlim(times[0], times[-1])
        ax.grid()
        place_r()
        fig.set_size_inches(6.15,2)
        fig.subplots_adjust(left=0.090, top=0.96,bottom=0.24, right=0.98)
        if save:
            fig.savefig(f'{directory}{title.replace(" ", "_")}.pdf')
        
        toc = timeit.default_timer()
        # print all the indicators for the run
        print("\n---------------------------------------------")
        print(f"Finished experiment: (elapsed time, {toc-tic:.4f} s)")
        print(f"    avg RSE warmup : {RSE_warmup:.3f}")
        print(f"    avg RSE flight : {RSE_flight:.3f}")
        print(f"    avg peak gs    : {max_abs_nz:.2f}")
        print(f"    avg Smoothness : {avg_Sm:.3f}")
        print("---------------------------------------------\n\n")
        if show:
            plt.show()
        else:
            plt.close('all')
