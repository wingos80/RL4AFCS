import matplotlib.pyplot as plt
import numpy as np
import time
import os
import scipy.stats as ss
from bisect import bisect_left
from typing import List
fig_scale = 1

NO_TRACE = None
A_TRACE  = 'accumulating'
R_TRACE  = 'replacing'

class COLOR:
   PURPLE = '\033[1;35;48m'
   CYAN = '\033[1;36;48m'
   BOLD = '\033[1;37;48m'
   BLUE = '\033[1;34;48m'
   GREEN = '\033[1;32;48m'
   YELLOW = '\033[1;33;48m'
   RED = '\033[1;31;48m'
   BLACK = '\033[1;30;48m'
   UNDERLINE = '\033[4;37;48m'
   END = '\033[1;37;0m'

def place_l():    mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(25,33,640,870)
def place_tl():   mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(25,33,640,500)
def place_bl():   mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(25,533,640,500)
def place_tr():   mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(665,33,640,500)
def place_r():    mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(665,282,640,500)
def place_br():   mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(665,533,640,500)
def place_trr():  mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(1280,33,640,500)
def place_brr():  mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(1280,533,640,500)
def place_smol(): mngr = plt.get_current_fig_manager(); mngr.window.setGeometry(1280,33,350,275)

def plotter(ax, x, y, ylabel, colors=None, log=False, legend=False):
    """
    Plotter function for general graphs
    
    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot the graph
    x : np.array
        x-axis vector
    y : dictionary
        List of length 2 in each value, 1st element is f(x),
        2nd element is transparency(alpha) of that graph,
        each key is the label for the corresponding graph
    ylabel : string
        Label for the y-axis
    colors : list, optional 
        List of colors for each plot, by default None
    log : bool, optional
        Plot in log scale, by default False
    legend : bool, optional
        Show legend, by default False
    """

    i=0
    for key, value in y.items():
        if colors:
            ax.plot(x, value[0], color=colors[i%len(colors)],label=key, alpha=value[1])
        else:
            ax.plot(x, value[0],label=key, alpha=value[1])
        i+=1
    if log:
        ax.set_yscale('log')
    if legend:
        ax.legend()
    ax.grid(True); ax.set_ylabel(ylabel)

def make_plots(x, ys, title, xlabel, ylabel, colors=None, save=False, log=[], legend=False):
    """
    Make multiple plots in one figure

    Parameters
    ----------    
    x : np.array
        x-axis vector    
    ys : list
        List of dictionaries, each dictionary is a plot        
    title : string
        Title of the plot        
    xlabel : string
        Label of the x-axis        
    ylabel : list
        List of labels for the y-axis of each subplots    
    colors : list, optional
        List of colors for each plot, by default None
    save : bool, optional
        Save the figure to a file, by default False        
    log : list, optional
        Index(s) of the plot(s) to be plotted in log scale, by default None    
    legend : bool, optional
        Show legend, by default False
    """
    figsize = (7.5*fig_scale, 5*fig_scale) if save else (6, 4)
    dpi     = 200 if save else 100
    fig, ax = plt.subplots(len(ys), 1, sharex=True, figsize=figsize, dpi=dpi)
    fig.canvas.manager.set_window_title(title) 
    plt.subplots_adjust(top=0.965, bottom=0.110, right=0.990, left=0.165)
    if len(ylabel) != len(ys):
        print('---Plot Warning, Number of ylabel must be equal to number of plots')
            

    if len(ys) == 1:
        ax = np.array([ax])
    
    ax[-1].set_xlabel(xlabel)

    for i, y in enumerate(ys):
        if i in log:
            plotter(ax[i], x, y, ylabel[i], colors=colors,log=True,legend=legend)
        else:
            plotter(ax[i], x, y, ylabel[i], colors=colors,log=None,legend=legend)

    plt.tight_layout()
    if save:
        plt.savefig(f'{title.replace(" ", "_")}.pdf')

    return fig, ax

def update_plots(x, ys, fig, axs):
    """
    Update the graph data in the figure

    Parameters
    ----------
    x : np.array
        x-axis vector
    ys : list
        List of dictionaries, each dictionary is a plot
    fig : matplotlib.figure
        Figure to be updated
    ax : matplotlib.axes
        Axis to be updated
    Returns
    -------
    fig : matplotlib.figure
        Updated figure
    ax : matplotlib.axes
        Updated axis
    """
    # TODO: add assert to make sure that number of ys and ls are same
    for i, ax in enumerate(axs):
        lines = list(ax.lines)
        y     = ys[i]
        for j, key in enumerate(y): # TODO: add assert that the label (key) is consistent?
            y_data_shape = y[key][0].shape
            if len(y_data_shape) == 1: # if there is only 1 line to plot in this entry
                lines[j].set_data(x, y[key][0])
            else:
                assert len(lines) == y_data_shape[-1], f'Number of lines in the plot must be same as the number of lines in the axes object, no. axes lines ({len(lines)}) != no. y_data lines{y_data_shape[-1]}'
                for k in range(y_data_shape[-1]):
                    lines[k].set_data(x, y[key][0][:,k])
                    

        axs[i].relim()
        axs[i].autoscale()
    
    return fig, axs

def get_PSD(t_end, dt, array):
    """
    Calculate the Power Spectral Density of an input signal

    Parameters
    ----------
    t_end : float
        End time of the signal in seconds
    dt : float
        Timestep of the signal in seconds
    array : np.array
        Input signal, shape=(n,) or (m,n) 
        where n is the number of signal samples 
        and m is the number of signals
    Returns
    -------
    spectra : np.array
        Power Spectral Density of the input signal
    omega : np.array
        Frequency range in Hz
    """
    
    fs    = 1/dt               # sampling frequency in Hz
    N     = int(t_end*fs)      # number of samples
    upp   = int(N/2)           # upper limit for the plot
    seq   = np.arange(0,upp,1) # sequence for the plot
    omega = seq/(N*dt)         # frequency range in Hz
    
    if len(array.shape) == 1:
        array = np.expand_dims(array, axis=0)
    
    n_signals = array.shape[0]
    array_out = np.zeros((n_signals,upp))
    for i in range(n_signals):
        array_fft = np.squeeze(array[i,:])
        array_fft = np.fft.fft(array_fft) # simply use numpy's fft func
        array_fft = array_fft*np.conjugate(array_fft)
        array_fft = 1/t_end*array_fft
        array_fft = array_fft[:upp]
        array_out[i] = np.abs(array_fft)

        # array_fft = np.squeeze(array[i,:])
        # array_fft = np.abs(np.fft.fft(array_fft)) # simply use numpy's fft func
        # array_fft = array_fft[:upp]
        # array_out[i] = array_fft
        # array_out[i] = array_fft/np.max(array_fft)
         
    spectra = np.squeeze(array_out)
    return spectra, omega

def pick_continuous_hparams(n_configs, lambda_hs=None, lambda_ls=None, lr_a_hs=None, lr_c_hs=None, lr_a_ls=None, lr_c_ls=None, kappas=None, cooldown_times=None, sigmas=None, warmup_times=None, elig_a=None, true_random=True):
    """
    Pick random values from a continuous range for each hyperparameter,
    each hyperparameter is a list of 2 elements, the first element is the 
    lower bound and the second element is the upper bound

    Returns
    ------
    dict
        Dictionary of hyperparameters, each key is the name of the hyperparameter,
        and each value is a list of n_configs elements 
    """
    if true_random:
        np.random.seed((os.getpid() * int(time.time()))%123456) # make sure randomly picking
    else:
        np.random.seed(0)

    lambda_hs      = np.random.uniform(lambda_hs[0], lambda_hs[1], n_configs) if lambda_hs else None
    lambda_ls      = np.random.uniform(lambda_ls[0], lambda_ls[1], n_configs) if lambda_ls else None
    lr_a_hs        = np.random.uniform(lr_a_hs[0], lr_a_hs[1], n_configs) if lr_a_hs else None 
    lr_c_hs        = np.random.uniform(lr_c_hs[0], lr_c_hs[1], n_configs) if lr_c_hs else None
    lr_a_ls        = np.random.uniform(lr_a_ls[0], lr_a_ls[1], n_configs) if lr_a_ls else None
    lr_c_ls        = np.random.uniform(lr_c_ls[0], lr_c_ls[1], n_configs) if lr_c_ls else None
    kappas         = np.random.uniform(kappas[0], kappas[1], n_configs) if kappas else None
    cooldown_times = np.random.uniform(cooldown_times[0], cooldown_times[1], n_configs) if cooldown_times else None 
    sigmas         = np.random.uniform( sigmas[0],  sigmas[1], n_configs) if sigmas else None    
    warmup_times   = np.random.uniform(warmup_times[0], warmup_times[1], n_configs) if warmup_times else None
    elig_a         = [elig_a for _ in range(n_configs)] if elig_a else None
    multistep      = [0 for _ in range(n_configs)]

    configs = {'multistep'      : multistep,
               'lambda_hs'      : lambda_hs,
               'lambda_ls'      : lambda_ls,
                'lr_a_hs'       : lr_a_hs,
                'lr_c_hs'       : lr_c_hs,
                'lr_a_ls'       : lr_a_ls,
                'lr_c_ls'       : lr_c_ls,
                'kappas'        : kappas,
                'cooldown_times': cooldown_times,
                'sigmas'        : sigmas,
                'warmup_times'  : warmup_times,
                'elig_a'        : elig_a
                }
    # round all entries to 3 significant figures
    for key, value in configs.items():
        if value is not None:
            if key == 'kappas':
                configs[key] = [int(v) for v in value]
            elif key == 'multistep':
                continue
            else:
                configs[key] = [round(v, 3) for v in value]

    return configs

def pick_discrete_hparams(n_configs, lambda_hs=None, lambda_ls=None, lr_a_hs=None, lr_c_hs=None, lr_a_ls=None, lr_c_ls=None, kappas=None, cooldown_times=None, sigmas=None, warmup_times=None, elig_a=None, lr_decays=None, true_random=True):
    # TODO make this a latin hypercube sampler
    """
    Sampling random values from a discrete range for each 
    hyperparameter, each hyperparameter is a list of n elements, where n is 
    the number of discrete values to pick from

    Returns
    ------
    dict
        Dictionary of hyperparameters, each key is the name of the hyperparameter,
        and each value is a list of n_configs elements 
    """
    if true_random:
        np.random.seed((os.getpid() * int(time.time()))%123456) # make sure randomly picking
    else:
        np.random.seed(0)

    lambda_hs      = np.random.choice(lambda_hs, n_configs) if lambda_hs else None
    lambda_ls      = np.random.choice(lambda_ls, n_configs) if lambda_ls else None
    lr_a_hs        = np.random.choice(lr_a_hs, n_configs) if lr_a_hs else None
    lr_c_hs        = np.random.choice(lr_c_hs, n_configs) if lr_c_hs else None
    lr_a_ls        = np.random.choice(lr_a_ls, n_configs) if lr_a_ls else None
    lr_c_ls        = np.random.choice(lr_c_ls, n_configs) if lr_c_ls else None
    kappas         = np.random.choice(kappas, n_configs) if kappas else None
    cooldown_times = np.random.choice(cooldown_times, n_configs) if cooldown_times else None
    sigmas         = np.random.choice(sigmas, n_configs) if sigmas else None
    warmup_times   = np.random.choice(warmup_times, n_configs) if warmup_times else None
    elig_a         = np.random.choice(elig_a, n_configs) if elig_a else None
    lr_decays      = np.random.choice(lr_decays, n_configs) if lr_decays else None
    seeds          = np.random.randint(0, 10000, n_configs)
    configs = {'lambda_hs'      : lambda_hs,
                'lambda_ls'     : lambda_ls,
                'lr_a_hs'       : lr_a_hs,
                'lr_c_hs'       : lr_c_hs,
                'lr_a_ls'       : lr_a_ls,
                'lr_c_ls'       : lr_c_ls,
                'kappas'        : kappas,
                'cooldown_times': cooldown_times,
                'sigmas'        : sigmas,
                'warmup_times'  : warmup_times,
                'elig_a'        : elig_a,
                'lr_decays'     : lr_decays,
                'seeds'         : seeds
                }
    
    # # clean out all the none values
    # config2 = {key: value for key, value in configs.items() if value is not None}
    # hparams = {f'hparam_{i}': [] for i in range(n_configs)}
    # for key, value in config2.items():
    #     for i, v in enumerate(value):
    #         hparams[f'hparam_{i}'].append(v)

    # check that no hparam list is repeated, if it is then latin sampling failed
    
    return configs

def get_convergence_time(c_hist, kappa, dt):
    """
    Find the time where angle of attack error is less than 0.5 degrees
    Parameters
    ----------
    c_hist : np.array
        History of the cost function
    kappa : float
        Reward scaling factor
    dt : float
        Timestep
    Returns
    -------
    float
        Convergence time in seconds
    """
    aoa_error = np.rad2deg(np.sqrt(-2*(c_hist/kappa)))
    converged_timestep = np.where(aoa_error > 0.5)[0][-1]
    converged_time     = converged_timestep*dt
    return converged_time

def get_all_used_hparams():
    """
    Get all the hyperparameters used in the experiments
    """
    main_dirr = 'IDHP/exps/nlin/hparams2/shift_cg/'
    kappas, etaahs, etaals, etachs, etacls, lambda_hs, lambda_ls = [], [], [], [], [], [], []

    for _, d in enumerate(os.listdir(main_dirr)):
        d = d.split('_')
        d = [''] + d
        # kappas.append(int(d[0].replace('k','')))
        etaahs.append(float(d[1].replace('etaah','')))
        etaals.append(float(d[2].replace('etaal','')))
        etachs.append(float(d[3].replace('etach','')))
        etacls.append(float(d[4].replace('etacl','')))
        lambda_hs.append(float(d[5].replace('lh','')))
        lambda_ls.append(float(d[6].replace('ll','')))
    
    return kappas, etaahs, etaals, etachs, etacls, lambda_hs, lambda_ls

def VD_A(X: List[float], Y: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    Parameters:
    variable: List[float]
        List of treatment values
    control: List[float]
        List of control values
    Returns:
    estimate: float
        Vargha and Delaney A index
    magnitude: str
        Magnitude of the effect size
    """
    m = len(X)
    n = len(Y)

    if m != n:
        Warning("Data 'variable' and 'control' should have the same length")
        length   = min(m, n) # truncate the longer list
        Y = Y[:length]
        X = X[:length]

    r  = ss.rankdata(Y + X)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels    = [0.06, 0.14, 0.21]  # effect sizes from Vargha and Delaney, 2000
    magnitude = [f"{COLOR.RED}negligible{COLOR.END}", f"{COLOR.RED}small{COLOR.END}", f"{COLOR.BLUE}medium{COLOR.END}", f"{COLOR.GREEN}large{COLOR.END}"]
    scaled_A  = A - 0.5

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate  = A

    return estimate, magnitude

def kl_divergence(u, v, epsilon=np.finfo(float).eps):
    """Kullback-Leibler divergence.

    Syonymes:
        KL divergence, relative entropy, information deviation

    References:
        1. Kullback S, Leibler RA (1951) On information and sufficiency.
           Ann. Math. Statist. 22:79–86
        2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
           Measures between Probability Density Functions. International
           Journal of Mathematical Models and Methods in Applied Sciences.
           1(4):300-307.
    """
    u = np.where(u<=0, epsilon, u)
    v = np.where(v<=0, epsilon, v)
    return np.sum(u * np.log(u / v))



if __name__ == '__main__':
    # a = np.array([1,2,3,4,5,6,7,8,9,10,1,124,51,6,25572,7,47316,2472,7])
    # b = a.copy()
    for i in [-1,-0.75,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,0.75,1]:
        rng = np.random.default_rng(2)
        a = rng.normal(10, 1, 100000)
        b = rng.normal(10+i, 1, 100000)
        # # shuffle a
        # rng.shuffle(a)
        # # shuffle b
        # rng.shuffle(b)
        kl_div = kl_divergence(a,b)
        a_val, mag = VD_A(a,b)
        # print(f'a:\n{a}\n\nb:\n{b}\n')
        print(f'{i} difference:')
        print(f'    KL divergence: {kl_div}')
        print(f'    A: {a_val}, magnitude: {mag}')
