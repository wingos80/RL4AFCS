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

import timeit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
# Force use CPU by hiding GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *


class Network(tf.keras.Model):
    """
    Generic network class for IDHP agent, used to create the actor and critic.
    Subclass of tf.keras.Model so that it can be used as a model with the
    tf.keras API.
    """

    def __init__(self, in_dim, layers, identity_init, std_init=0.01, seed=1):
        """
        Constructor for the Network class.
        Parameters:
        ----------
        in_dim : int
            Dimension of the input layer.
        layers : dict
            Dictionary with each entry being one layer of the network. The key
            is the layer size and the value is the activation function. 
            Note the output layer is also defined in this dictionary!
        identity_init : bool
            If true, the network is initialized to produce a 1 to 1 mapping between input output.
        std_init : float
            Standard deviation of the normal distribution used for initializing the network weights.
        seed : int
            Random seed used for initializing the network weights.
        """
        super().__init__()
        # Dimensions
        self.input_dim = in_dim

        # Network
        self.network = None
        self.layers_dict = layers
        if identity_init:
            self.kernel_initializer = tf.keras.initializers.Identity()
        else:
            self.kernel_initializer = tf.keras.initializers.truncated_normal(stddev=std_init, seed=seed)

        self.n_layers = len(layers)
        self.xi       = [None for i in range(self.n_layers)] # list to store layer inputs
        self.ai       = [None for i in range(self.n_layers)] # list to store activation func derivatives
        
        # Create the network
        self.setup()

    def setup(self):
        """
        Build the neural network model
        """
        # Instantiate a sequential model
        self.network=tf.keras.Sequential()
        self.network.add(tf.keras.Input(shape=(self.input_dim,)))

        in_dim   = self.input_dim
        n_params = 0
        # Hidden layers and final otput layer
        for key, value in self.layers_dict.items():
            layer_size, activation = key, value
            assert activation in ['tanh', 'linear'], f"Activation function {activation} not supported, use 'tanh' or 'linear'"

            self.network.add(tf.keras.layers.Dense(
                layer_size,
                activation=activation,
                use_bias=False,
                kernel_initializer=self.kernel_initializer)
            )

            n_params += in_dim*layer_size
            in_dim = layer_size

        n_outputs = layer_size
        self.E = np.zeros((n_outputs, n_params)) # eligibility trace, size of n_outputs times n_params
    
    def base_call(self, s):
        """
        Give the network output, using own implementation of forward pass to enable eligibility trace updates
        """
        # s = self.network(s) # tf implementation of feedforward

        s = s
        i = 0
        # Forward pass s through the network
        for _, value in self.layers_dict.items():
            # store input to each layer
            self.xi[i] = s.numpy()
            # pass input through layer weights
            s = s@self.network.layers[i].weights[0] # have to use this stupid ass way to do this, tf.network(s) will skip all the variables

            # pass through layer activation function
            s = self.network.layers[i].activation(s)
            
            # store activation function derivative
            if value == 'tanh':
                self.ai[i] = 1 - s.numpy()**2 # in np.array
            elif value == 'linear':
                self.ai[i] = np.ones_like(s)  # in np.array
            else:
                raise ValueError(f"Activation function {self.layers_dict[i+1]} not supported, use 'tanh' or 'linear'")
            
            i += 1
        
        return s
        # return self.network(s)

class Critic(Network):
    """
    Critic class for IDHP agent, estimates value function gradients.
    """
    def __init__(self, in_dim, layers, identity_init, std_init=0.01, seed=1, eligibility=None):
        super().__init__(in_dim, layers, identity_init, std_init, seed=seed)
        self.network.output_names = "Lambda"
        self.eligibility = eligibility

    def call(self, s):
        """
        Call the critic network
        """
        s = self.base_call(s)

        # W1 = self.trainable_weights[0].numpy().T
        W2 = self.trainable_weights[1].numpy().T

        # TODO indicies are hard coded for critic of (1x4x2)
        if self.eligibility is None:
            temp = self.xi[1].flatten()
            self.E[0,0:4]  = temp
            self.E[0,8:12] = (W2[0,:]*self.ai[0]*self.xi[0]).flatten()
            self.E[1,4:8]  = temp
            self.E[1,8:12] = (W2[1,:]*self.ai[0]*self.xi[0]).flatten()

        elif self.eligibility == 'accumulating':
            self.E *= self.gamma_lambda if self.eligibility else 0

            temp = self.xi[1].flatten()
            self.E[0,0:4]  += temp
            self.E[0,8:12] += (W2[0,:]*self.ai[0]*self.xi[0]).flatten()
            self.E[1,4:8]  += temp
            self.E[1,8:12] += (W2[1,:]*self.ai[0]*self.xi[0]).flatten()

        elif self.eligibility == 'replacing':
            self.grad = np.zeros_like(self.E)
            temp = self.xi[1].flatten()
            self.grad[0,0:4] = temp
            self.grad[0,8:12] = (W2[0,:]*self.ai[0]*self.xi[0]).flatten()
            self.grad[1,4:8] = temp
            self.grad[1,8:12] = (W2[1,:]*self.ai[0]*self.xi[0]).flatten()

            if np.linalg.norm(self.grad) > np.linalg.norm(self.E):
                self.E = self.grad
            else:
                self.E *= self.gamma_lambda

        # # bound all elements of E between -1 and 1
        # self.E = np.clip(self.E, -1, 1)

        return s
    
    def get_weight_update(self, td_error):
        """
        Get the weight update for the critic network
        """
        # get the gradients of the loss wrt the network weights
        # with respect to the last layer
        critic_grad = td_error@self.E

        W2_update = tf.transpose(tf.reshape(critic_grad[0, 0:8],(2,4)))
        W1_update = tf.transpose(tf.reshape(critic_grad[0,8:12],(4,1)))
        return [W1_update, W2_update]

    def soft_update(self, source_weights, tau):
        """
        Update parameters from a source network
        """
        source = source_weights
        target = self.trainable_weights

        for target_var, source_var in zip(target, source):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)

class Actor(Network):
    """
    Critic class for IDHP agent, estimates policy function.
    """
    def __init__(self, in_dim, layers, identity_init, std_init=0.01, seed=1, eligibility=None):
        super().__init__(in_dim, layers, identity_init, std_init=std_init, seed=seed)
        self.network.output_names = "Actions"
        self.eligibility = eligibility
         
    def call(self, s):
        """
        Call the actor network
        """
        s = self.base_call(s)

        # W1 = self.trainable_weights[0].numpy().T
        W2 = self.trainable_weights[1].numpy().T

        # TODO indicies are hard coded for actor of (1x4x1)
        if self.eligibility is None:
            self.E[0,:4] = (self.ai[1]@self.xi[1]).flatten()
            self.E[0,4:] = ((self.ai[1]@W2*self.ai[0]).T@self.xi[0]).flatten()
        
        elif self.eligibility == 'accumulating':
            self.E *= self.gamma_lambda

            self.E[0,:4] += (self.ai[1]@self.xi[1]).flatten()
            self.E[0,4:] += ((self.ai[1]@W2*self.ai[0]).T@self.xi[0]).flatten()

        elif self.eligibility == 'replacing':
            self.grad = np.zeros_like(self.E)
            self.grad[0,:4] = (self.ai[1]@self.xi[1]).flatten()
            self.grad[0,4:] = ((self.ai[1]@W2*self.ai[0]).T@self.xi[0]).flatten()
            
            if np.linalg.norm(self.grad) > np.linalg.norm(self.E):
                self.E = self.grad
            else:
                self.E *= self.gamma_lambda if self.eligibility else 0

        # bound all elements of E between -1 and 1
        # print(np.linalg.norm(self.E))
        # self.E = np.clip(self.E, -1, 1)
        return s
    
    def get_weight_update(self, loss):
        """
        Get the weight update for the critic network
        """
        # get the gradients of the loss wrt the network weights
        # with respect to the last layer
        actor_grad = loss@self.E

        W2_update = tf.reshape(actor_grad[0,0:4],(4,1))
        W1_update = tf.reshape(actor_grad[0,4:8],(1,4))
        return [W1_update, W2_update]

    def soft_update(self, source_weights, tau):
        """
        Update parameters from a source network
        """
        source = source_weights
        target = self.trainable_weights

        for target_var, source_var in zip(target, source):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)

class Critic_big(Network):
    """
    Critic class for IDHP agent, estimates value function gradients.
    The only difference with Critic() is that the eligibility
    trace sizes are different.
    """
    def __init__(self, in_dim, layers, identity_init, std_init=0.01, seed=1, eligibility=None):
        super().__init__(in_dim, layers, identity_init, std_init, seed=seed)
        self.network.output_names = "Lambda"
        self.eligibility = eligibility

    def call(self, s):
        """
        Call the critic network
        """
        s = self.base_call(s)

        # W1 = self.trainable_weights[0].numpy().T
        W2 = self.trainable_weights[1].numpy().T

        # TODO indicies are hard coded for critic of (4x10x3)
        if self.eligibility is None:
            return s
            temp = self.xi[1].flatten()
            self.E[0,0:4]  = temp
            self.E[0,8:12] = (W2[0,:]*self.ai[0]*self.xi[0]).flatten()
            self.E[1,4:8]  = temp
            self.E[1,8:12] = (W2[1,:]*self.ai[0]*self.xi[0]).flatten()

        elif self.eligibility == 'accumulating':
            raise('Not implemented')
            self.E *= self.gamma_lambda if self.eligibility else 0

            temp = self.xi[1].flatten()
            self.E[0,0:4]  += temp
            self.E[0,8:12] += (W2[0,:]*self.ai[0]*self.xi[0]).flatten()
            self.E[1,4:8]  += temp
            self.E[1,8:12] += (W2[1,:]*self.ai[0]*self.xi[0]).flatten()

        elif self.eligibility == 'replacing':
            raise('Not implemented')
            self.grad = np.zeros_like(self.E)
            temp = self.xi[1].flatten()
            self.grad[0,0:4] = temp
            self.grad[0,8:12] = (W2[0,:]*self.ai[0]*self.xi[0]).flatten()
            self.grad[1,4:8] = temp
            self.grad[1,8:12] = (W2[1,:]*self.ai[0]*self.xi[0]).flatten()

            if np.linalg.norm(self.grad) > np.linalg.norm(self.E):
                self.E = self.grad
            else:
                self.E *= self.gamma_lambda

        # # bound all elements of E between -1 and 1
        # self.E = np.clip(self.E, -1, 1)

        return s
    
    def get_weight_update(self, td_error):
        """
        Get the weight update for the critic network
        """
        # get the gradients of the loss wrt the network weights
        # with respect to the last layer
        critic_grad = td_error@self.E

        W2_update = tf.transpose(tf.reshape(critic_grad[0, 0:8],(2,4)))
        W1_update = tf.transpose(tf.reshape(critic_grad[0,8:12],(4,1)))
        return [W1_update, W2_update]

    def soft_update(self, source_weights, tau):
        """
        Update parameters from a source network
        """
        source = source_weights
        target = self.trainable_weights

        for target_var, source_var in zip(target, source):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)

class Actor_big(Network):
    """
    Critic class for IDHP agent, estimates policy function. 
    The only difference with Actor() is that the eligibility
    trace sizes are different.
    """
    def __init__(self, in_dim, layers, identity_init, std_init=0.01, seed=1, eligibility=None):
        super().__init__(in_dim, layers, identity_init, std_init=std_init, seed=seed)
        self.network.output_names = "Actions"
        self.eligibility = eligibility
         
    def call(self, s, trace=True):
        """
        Call the actor network
        """
        s = self.base_call(s)

        W2 = self.trainable_weights[1].numpy().T

        if trace:
            # TODO indicies are hard coded for actor of (4x10x1)
            if self.eligibility is None:
                self.E[0,:10] = (self.ai[1]@self.xi[1]).flatten()
                self.E[0,10:] = ((self.ai[1]@W2*self.ai[0]).T@self.xi[0]).flatten()
            
            elif self.eligibility == 'accumulating':
                self.E *= self.gamma_lambda

                self.E[0,:10] += (self.ai[1]@self.xi[1]).flatten()
                self.E[0,10:] += ((self.ai[1]@W2*self.ai[0]).T@self.xi[0]).flatten()

            elif self.eligibility == 'replacing':
                self.grad = np.zeros_like(self.E)
                self.grad[0,:10] = (self.ai[1]@self.xi[1]).flatten()
                self.grad[0,10:] = ((self.ai[1]@W2*self.ai[0]).T@self.xi[0]).flatten()
                
                if np.linalg.norm(self.grad) > np.linalg.norm(self.E):
                    self.E = self.grad
                else:
                    self.E *= self.gamma_lambda if self.eligibility else 0

            # bound all elements of E between -1 and 1
            # print(np.linalg.norm(self.E))
            # self.E = np.clip(self.E, -1, 1)
        return s
        
    def update_traces(self, jacobian):
        """
        Update the eligibility trace for the actor network
        """
        self.E *= self.gamma_lambda
        self.E[0, :10] = jacobian[1].numpy().flatten()
        self.E[0, 10:] = jacobian[0].numpy().flatten()

    def get_weight_update(self, loss):
        """
        Get the weight update for the critic network
        """
        # get the gradients of the loss wrt the network weights
        # with respect to the last layer
        actor_grad = loss@self.E

        W2_update = tf.reshape(actor_grad[0,0:10],(10,1))
        W1_update = tf.transpose(tf.reshape(actor_grad[0,10:],(10,4)))
        return [W1_update, W2_update]

    def soft_update(self, source_weights, tau):
        """
        Update parameters from a source network
        """
        source = source_weights
        target = self.trainable_weights

        for target_var, source_var in zip(target, source):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)

class RLS():
    """
    Exponentially Weighted Recursive Least Squares system identification from Simon S. Haykin, 
    for use in IDHP.
    """
    def __init__(self, config) -> None:
        # TODO: docstring
        
        # store state and action space dimensions
        self.state_dim  = config['state_dim']
        self.action_dim = config['action_dim']
        self.gamma      = config['rls_gamma']
        self.init_cov   = config['rls_cov']

        self.A = np.array([[-0.73906365,  0.97442344],
                           [-1.47226503, -1.56667814]])

        self.B = np.array([[0.0008416],
                           [0.02884]])
        
        # params : [F.T; 
        #           G.T]
        self.params = np.zeros((self.state_dim + self.action_dim, self.state_dim))
        # self.params[0:2,0:2] = (np.eye(2) + 0.02*self.A).T
        # self.params = np.array([[ 9.85218727e-01,  1.94884688e-02, -6.23756182e-04],
        # [-2.94453116e-02,  9.68666437e-01, -4.69290444e-02]]).T
        # self.params[2,0:2] = 0.02*self.B.T
        
        # self.params = np.array([[ 9.90994818e-01, 1.98913143e-02, -3.21080803e-04],
        #                         [-7.59788294e-02, 9.89933184e-01, -1.15397381e-02]]).T
        # print(self.params.T)
        self.Cov    = self.init_cov*np.eye(self.state_dim + self.action_dim)

        # need to initialize non empty storage to avoid plotting error, since RLS is only updated after 1st 2 timesteps
        self.eps_norm_hist = [0,0]
        self.eps_hist      = [np.zeros(self.state_dim)]
        self.covs          = [self.Cov.flatten(), self.Cov.flatten()]
    
    def _reset(self):
        """
        Reset the RLS estimator
        """
        self.params = np.zeros((self.state_dim + self.action_dim, self.state_dim))
        self.Cov    = self.init_cov*np.eye(self.state_dim + self.action_dim)

    @property
    def F(self):
        return np.array(self.params[:self.state_dim, :].T)

    @property
    def G(self):
        return np.array(self.params[self.state_dim:, :].T)

    def update(self, dx_t, da_t, dx_t1):
        """
        Update the RLS estimator usng new observations
        Parameters:
        ----------
        dx_t : np.ndarray, shape=(n, 1)
            State increment at time t, defined as subtraction of x(t) - x(t-1)
        da_t : np.ndarray, shape=(m, 1)
            Action increment at time t, defined as subtraction of a(t) - a(t-1)
        dx_t1 : np.ndarray, shape=(n, 1)
            State increment at time t+1, defined as subtraction of x(t+1) - x(t)
        """
        ## Retreiving current RLS parameters and covariance
        params_t = self.params
        Cov_t    = self.Cov

        ## Constructing the state-action vector
        # X = [s;
        #      a]
        X_t       = np.concatenate((dx_t, da_t),axis=0)
        # self._asserts(X_t)

        ## predicting the x increment using current time step's RLS parameters
        dx_t_pred = params_t.T@X_t
        epsilon    = dx_t1 - dx_t_pred

        ## update the RLS parameters
        CX        = Cov_t@X_t
        XtCX      = X_t.T@CX
        Kt        = CX/(self.gamma + XtCX)
        params_t1 = params_t + Kt@epsilon.T

        ## update the covariance

        # next step should technically be Cov@X@X.T@Cov = CX@X.T@Cov, but assuming symmetric Cov, 
        # Cov.T = Cov, therefore: Cov@X@X.T@Cov = Cov@X@X.T@Cov.T = CX@CX.T
        # Equivalently, the version implemented below expresses the terms in the Kt gain term
        Cov_diff = Cov_t - Kt@CX.T
        Cov_t1   = Cov_diff/self.gamma

        ## Update the RLS parameters and covariance
        self.params = params_t1
        self.Cov    = Cov_t1
        
        ## Store the norm of the epsilon vector
        self.epsilon  = epsilon
        self.eps_hist.append(epsilon.flatten())
        self.eps_norm = np.linalg.norm(epsilon)
        self.eps_norm_hist.append(self.eps_norm)
        self.covs.append(self.Cov.flatten())

        tte = 1

    def _asserts(self, Xt):
        """
        Asserts for the RLS update
        """
        assert Xt.shape == (self.state_dim + self.action_dim, 1), f"incorrect input shape, Xt shape: {Xt.shape}, expected shape: {(self.state_dim + self.action_dim, 1)}"

class IDHPsp():
    def __init__(self, env, config, verbose=True, seed=1) -> None:
        # set seed for tensorflow
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

        self.seed  = seed
        self.gamma = config['gamma']
        self.tau   = config['tau']
        self.ms    = True if config['multistep']>0 else False

        # time at which to switch from high to low learning rates
        self.warmup_time  = config['warmup_time']
        self.error_thresh = config['error_thresh']  # error threshold in degrees (currently for alpha!)

        # adaptive switches
        self.changed         = False
        self.cooldown_1      = 0
        self.cooldown_timeit = int(config['cooldown_time']/env.dt) # 1.5 second cooldown for changing high low learning rates

        # store environment
        self.env       = env
        self.env.kappa = config['kappa']
        
        # setup the networks for the short period model
        self._setup_networks(config, seed)

        # debugging logs
        self.verbose = verbose
    
    def _setup_networks(self, config, seed):
        """
        Setup the IDHP agent for controlling the short period model 
        """
        # setting up actor and critic
        actor_layers    = config['actor_config']['layers']
        critic_layers   = config['critic_config']['layers']
        self.hidden_dim = list(config['actor_config']['layers'].keys())[0] # TODO, ugly, fix this

        self.actor         = Actor(config['in_dims'], actor_layers, False, config['sigma'], seed, config['actor_config']['elig'])
        self.critic        = Critic(config['in_dims'], critic_layers, False, config['sigma'], seed, config['critic_config']['elig'])
        self.target_critic = Critic(config['in_dims'], critic_layers, False, config['sigma'], seed, config['critic_config']['elig']) # does not matter how this is initialized, it'll be reset to be equal to critic

        # # experiment with output of one side of the network to zero
        # weights = self.critic.get_weights()
        # weights[-1][:,1] = np.zeros_like(weights[-1][:,1])
        # self.critic.set_weights(weights)
        # self.target_critic.set_weights(weights)


        self.lambda_h, self.lambda_l    = config['lambda_h'], config['lambda_l'] 
        gamma_lambda                    = self.gamma*self.lambda_h
        self.actor.gamma_lambda         = gamma_lambda
        self.critic.gamma_lambda        = gamma_lambda
        self.target_critic.gamma_lambda = gamma_lambda
        
        self.eta_a_h = config['actor_config']['eta_h']
        self.eta_c_h = config['critic_config']['eta_h']
        self.eta_a_l = config['actor_config']['eta_l']
        self.eta_c_l = config['critic_config']['eta_l']
        
        # setting up rls model
        self.model = RLS(config['rls_config'])
        self.n     = self.model.state_dim
        self.m     = self.model.action_dim

    def _reset_logs(self, N, o):
        """
        Reset all the logs
        """
        n, m = self.n, self.m
        self.t_hist   = np.zeros(N)
        self.x_hist   = np.zeros((N,n))
        self.a_hist   = np.zeros((N,m))
        self.s_hist   = np.zeros((N,o))
        self.c_hist   = np.zeros(N)
        self.ref_hist = np.zeros(N)

        self.a_weights_hist1 = np.zeros((N,self.hidden_dim))
        self.a_weights_hist2 = np.zeros((N,self.hidden_dim))
        self.c_weights_hist1 = np.zeros((N,self.hidden_dim))
        # TODO hardcoded for 1x4x2 critic network 
        self.c_weights_hist2 = np.zeros((N,self.hidden_dim*2)) 

        self.a_gain     = np.zeros((N,30))

        # TODO hardcoded for 1x4x2 critic network and 1x4x1 actor network
        self.a_all_grad_hist = np.zeros((N,8))
        self.a_e_hist        = np.zeros((N,8))
        self.c_all_grad_hist = np.zeros((N,12))
        self.c_e_hist        = np.zeros((N,24))

        self.a_grad_hist = np.zeros((N))
        self.c_grad_hist = np.zeros((N))

        self.params_hist   = np.zeros((N,n*(n+m)))  # store params in flattened array
        self.cov_hist      = np.zeros((N,(n+m)**2)) # store cov in flattened array
        self.eps_norm_hist = np.zeros((N))
        self.eps_hist      = np.zeros((N,n))

    def _log(self, i, t, s, a, x, c):
        """
        Log the current state, action, and reward
        """
        # if any is nan, set every subsequent data point as nan
        if np.isnan(s).any() or np.isnan(a).any() or np.isnan(x).any() or np.isnan(c):
            self.t_hist[i:]   = np.nan
            self.a_hist[i:]   = np.nan
            self.s_hist[i:]   = np.nan
            self.x_hist[i:]   = np.nan
            self.c_hist[i:]   = np.nan
            self.ref_hist[i:] = np.nan

            self.a_weights_hist1[i:] = np.nan
            self.a_weights_hist2[i:] = np.nan
            self.c_weights_hist1[i:] = np.nan
            self.c_weights_hist2[i:] = np.nan

            self.a_all_grad_hist[i:] = np.nan
            self.a_e_hist[i:]        = np.nan
            self.c_all_grad_hist[i:] = np.nan
            self.c_e_hist[i:]        = np.nan

            self.a_grad_hist[i:] = np.nan
            self.c_grad_hist[i:] = np.nan

            self.params_hist[i:] = np.nan
            self.cov_hist[i:]    = np.nan
            self.eps_norm_hist[i:] = np.nan
            

        self.t_hist[i]   = t
        self.a_hist[i]   = a
        self.s_hist[i]   = s.numpy().flatten().copy() # convert to numpy and flatten
        self.x_hist[i,:] = x.T
        # self.c_hist[i]   = np.rad2deg((c*-2/28)**(0.5))
        self.c_hist[i]   = c
        self.ref_hist[i] = self.env.yref_hist[i]

        # TODO hardcoded for 1x4x2 critic network and 1x4x1 actor network
        self.a_weights_hist1[i,:] = self.actor.trainable_weights[0].numpy()[0,:]
        self.a_weights_hist2[i,:] = self.actor.trainable_weights[-1].numpy().T
        self.c_weights_hist1[i,:] = self.critic.trainable_weights[0].numpy()[0,:]
        self.c_weights_hist2[i,:] = self.critic.trainable_weights[-1].numpy().flatten()

        # self.temp.soft_update(self.actor.trainable_weights, 1)
        # for j in range(30):
        #     inp = np.deg2rad(float((j-15)/3))
        #     self.a_gain[i,j] = 20*self.temp(tf.constant([[inp]]))

        self.c_e_hist[i,:]        = np.array(self.critic.E).flatten()
        self.a_e_hist[i,:]        = np.array(self.actor.E).flatten()

        if i > 1:
            # store the gradient norms
            self.a_grad_hist[i] = np.linalg.norm(self.actor_loss_grad[0].numpy().flatten())
            self.c_grad_hist[i] = np.linalg.norm(self.critic_loss_grad[0].numpy().flatten())

            # TODO hardcoded for 1x4x1 actor network
            # store all the gradients
            temp_a = np.zeros(8)
            temp_a[:4] = self.actor_loss_grad[0].numpy().flatten()
            temp_a[4:] = self.actor_loss_grad[1].numpy().flatten()
            self.a_all_grad_hist[i,:] = temp_a

            # TODO hardcoded for 1x4x2 critic network
            temp_c = np.zeros(12)
            temp_c[:4] = self.critic_loss_grad[0].numpy().flatten()
            temp_c[4:] = self.critic_loss_grad[1].numpy().flatten()
            self.c_all_grad_hist[i,:] = temp_c

            # store the RLS parameters and covariance
            self.params_hist[i,:] = self.model.params.flatten().copy()
            self.cov_hist[i,:]    = self.model.Cov.flatten().copy()
            self.eps_norm_hist[i] = self.model.eps_norm
            self.eps_hist[i,:]    = np.abs(self.model.epsilon.flatten().copy())

    def _info(self, step):
        window = int(5/self.env.dt)
        window = window if step > window else step
        
        lb, ub = step-window, step
        x   = self.x_hist[lb:ub, :]
        ref = self.env.yref_hist[lb:ub]
        a   = self.a_hist[lb:ub, :]
        t   = self.t_hist[lb:ub]

        
        if step == 0:
            # plot alpha and q along with the reference in alpha
            # TODO remove hard coding for these y labels
            y_alpha = {r'$\delta \alpha$'      : [np.rad2deg(x[0,0]), 1],
                       r'$\delta \alpha_{ref}$': [np.rad2deg(ref[0]), 1]}
            y_q     = {r'$\delta q$'           : [np.rad2deg(x[0,1]), 1]}
            # TODO hard coded action scaling
            y_de    = {r'$\delta d_e$'         : [20*a, 1]}

            self.fig2, self.ax2 = make_plots(t, [y_alpha, y_q, y_de], f'States and actions', r'Time $[s]$', [r'$\delta \alpha$', r'$\delta q$', r'$\delta d_e$'], legend=True)
        
            # put window on top right
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(665,33,640,500)
        else:
            y_alpha = {r'$\delta \alpha$'      : [np.rad2deg(x[:, 0]), 1], 
                       r'$\delta \alpha_{ref}$': [np.rad2deg(ref[:]), 1]}
            y_q     = {r'$\delta q$'           : [np.rad2deg(x[:, 1]), 1]}
            y_de    = {r'$\delta d_e$'         : [20*a, 1]}             
            self.fig2, self.ax2 = update_plots(t, [y_alpha, y_q, y_de], self.fig2, self.ax2)

            plt.pause(0.02)

    def _get_network_input(self, s,e):
        """
        Get the input for the network, useful for desining network inputs
        Mainly converts the inputs to a tensorflow tensor object
        """
        # temp = tf.concat([s, [[e]]], axis=0)
        # temp = tf.reshape(temp,(1,3))
        temp = tf.constant([[e]], dtype=tf.float32)
        # temp = tf.reshape(s,(1,2))
        return temp

    def _asserts(self, nn_in, info, a):
        """
        Asserts for the IDHP update
        """
        c_grad   = info['reward_grad'].copy()
        lambda_0 = self.critic(nn_in)

        assert a.shape[0]     == self.m, f"Action shape must match action dimension, action shape: {a.shape}, action dimension: {self.m}"
        assert lambda_0.shape == c_grad.shape, f"Lambda and c_grad shape must match, lambda_0 shape: {lambda_0.shape}, c_grad shape: {c_grad.shape}"

    def _adapt_check(self, step, info):
        """
        Adapt the learning rate
        Parameters:
        ----------
        step : int
            Current timestep
        """
        error_threshold = np.deg2rad(self.error_thresh)                 # convert to radians
        epsilon_norm_threshold = 5e-5                                   # rls model reset threshold on model residual
        condition_1 = step < int(self.warmup_time/self.env.dt)          # true if simulation in first time_c seconds
        condition_2 = np.abs(info['e']) > error_threshold               # true if error is above threshold
        condition_3 = self.model.eps_norm > epsilon_norm_threshold      # true if epsilon is above threshold

        # # measure the average tracking error of the last 50 timesteps
        # if step < 50:
        #     lb = 0
        # else:
        #     lb = step - 50
        # aoa_error = self.c_hist[lb:step]/self.env.kappa
        # avg_err = np.average(np.abs(aoa_error))

        # condition_4 = avg_err > 0.5      # true if average error is above 0.5 degrees


        # decay cooldown for changing learning rates
        if self.cooldown_1 > 0:
            self.cooldown_1 -= 1
        # print(f"step: {step}, etac: {self.eta_c}, etaa: {self.eta_a}, cooldown: {self.cooldown_1}, condition1 {condition_1}, condition2 {condition_2}, condition3 {condition_3}")
        if condition_1 or condition_2:
            eta_a, eta_c = np.array(self.eta_a_h, np.float32), np.array(self.eta_c_h, np.float32)
            lambda_gamma = self.lambda_h*self.gamma
        else:
            eta_a, eta_c = np.array(self.eta_a_l, np.float32), np.array(self.eta_c_l, np.float32)
            lambda_gamma = self.lambda_l*self.gamma
        
        if eta_a != self.eta_a and self.cooldown_1 == 0:
            # print(f'\nadapting parameters at time {step*self.env.dt}s')
            # print(f'eta_a: {self.eta_a} -> {eta_a}')
            # print(f'eta_c: {self.eta_c} -> {eta_c}')
            # print(f'lambda_gamma: {self.actor.gamma_lambda} -> {lambda_gamma}')

            self.eta_a = eta_a
            self.actor_optimizer.learning_rate.assign(self.eta_a)

            self.eta_c = eta_c
            self.critic_optimizer.learning_rate.assign(self.eta_c)

            self.actor.gamma_lambda         = lambda_gamma
            self.critic.gamma_lambda        = lambda_gamma

            # changed learning rate, so start cooldown counter
            self.cooldown_1 = self.cooldown_timeit

        
        if (condition_3 and not self.changed) and not condition_1:
            # print(f"RLS reset, time: {step*self.env.dt}")
            self.model._reset()
            self.changed = True

    def _invert_controller(self):
        """
        function to flip sign of the actor network
        """
        dummy_input = self._get_network_input(tf.constant([[1.0]], dtype=tf.float32), 1.0)

        # print(f"inverting controller, previous output: {self.actor(dummy_input)}")
        self.actor.trainable_weights[-1].assign(-self.actor.trainable_weights[-1])
        # print(f"inverted controller, new output: {self.actor(dummy_input)}")

    def _step_networks(self, s_prev, s_next, s, e):
        with tf.GradientTape(persistent=True) as tape:
            # Get input state for networks and watch it for getting dadx
            nn_in_0 = self._get_network_input(s_prev, e)
            tape.watch(nn_in_0)
            if self.ms:
                nn_in_n = self._get_network_input(s_next, e)
                tape.watch(nn_in_n)
            else:
                nn_in_n = self._get_network_input(s, e)
                tape.watch(nn_in_n)

            # Get value function gradients (lambda)
            lambda_0  = self.critic(nn_in_0)
            lambdap_n = self.target_critic(nn_in_n)

            # Get input state for actor
            a_temp       = self.actor(nn_in_0)
            action_nodes = tf.split(a_temp, self.m, axis=1)

        a_next = a_temp.numpy().copy()
        F, G   = self.model.F, self.model.G
        
        grad = []
        for node in action_nodes:
            grad.append(tape.gradient(node, nn_in_0))
        
        

        return a_next, lambda_0, lambdap_n, F, G, grad, tape, a_temp
    
    def _update_networks(self, lambda_0, lambdap_n, c_grad, dx1dx0, dx1dx0_prev, c_grad_prev, G, tape, a_temp):
        # update critic
        if self.ms:
            td_error = lambda_0 - c_grad_prev - self.gamma*c_grad @ dx1dx0_prev - (self.gamma**2)*lambdap_n @ dx1dx0_prev @ dx1dx0
        else:
            td_error = lambda_0 - c_grad - self.gamma*lambdap_n @ dx1dx0
            
        self.critic_loss_grad = self.critic.get_weight_update(td_error)
        self.critic_optimizer.apply_gradients(zip(self.critic_loss_grad, self.critic.trainable_weights))

        # update target critic
        self.target_critic.soft_update(self.critic.trainable_weights, tau=self.tau)

        # # experiment with output of one side of the network to zero
        # weights = self.critic.get_weights()
        # weights[-1][:,1] = np.zeros_like(weights[-1][:,1])
        # self.critic.set_weights(weights)
        # self.target_critic.set_weights(weights)
        
        # update actor
        loss_grad        = (c_grad + self.gamma*lambdap_n) @ G # c_grad techinically shouldnt exists, but with smoothness assumption this works
        
        self.actor_loss_grad  = self.actor.get_weight_update(loss_grad)
        a_loss_grad = tape.gradient(a_temp, self.actor.trainable_weights, output_gradients=loss_grad)
        self.actor_optimizer.apply_gradients(zip(self.actor_loss_grad, self.actor.trainable_weights))


    def _setup_optimizers(self):
        """
        Setup the optimizers for idhp using the short period model
        """
        self.eta_a = self.eta_a_h
        self.eta_c = self.eta_c_h

        self.actor_optimizer  = tf.keras.optimizers.SGD(learning_rate=self.eta_a)
        self.critic_optimizer = tf.keras.optimizers.SGD(learning_rate=self.eta_c)
    
    def train(self):
        """
        Train the IDHP agent using multi step method
        """
        # initialize target critic to be identical to critic
        self.target_critic.soft_update(self.critic.trainable_weights, tau=1)
        
        # initialize all the loop variables
        t, dt, t_end           = 0, self.env.dt, self.env.t_end
        steps                  = int(t_end/dt)
        s, c, done, _, info    = self.env.reset(seed=self.seed)
        nn_in                  = self._get_network_input(s, info['e'])
        a                      = self.actor(nn_in).numpy()
        x                      = info['x']
        s_prev, a_prev, x_prev = None, None, None

        # multistep update variables
        c_grad, c_grad_prev = info['reward_grad'].copy(), None
        dx1dx0_prev = None


        o = s.shape[0] # TODO: maybe add assert for making sure s is a vector
        self._reset_logs(steps, o)

        self._setup_optimizers()

        self._asserts(nn_in, info, a)
        self._invert_controller()

        for step in (bar := tqdm(range(steps))):
            bar.set_description("   IDHP wowowow? pls work")
            
            t      = dt*step
            # execute action and observe new state, 
            s_next, c, done, _, info = self.env.step(20*a)
            s_next = tf.convert_to_tensor(s_next, dtype=tf.float32)
            x_next = info['x'].copy()
            e      = info['e'].copy()
            c_grad = info['reward_grad'].copy()

            a_next, lambda_0, lambdap_n, F, G, grad, tape, a_temp = self._step_networks(s_prev, s_next, s, e)

            dadx   = tf.reshape(grad, (1, self.m)) 
            dx1dx0 = F + G @ dadx
            # RLS update rule requires 2 past timestep's states and actions, for convenience RLS & networks 
            # are updated together after 0th and 1st timestep (0th time step taken before loop starts)
            if step > 0:
                # update networks
                self._update_networks(lambda_0, lambdap_n, c_grad, dx1dx0, dx1dx0_prev, c_grad_prev, G, tape, a_temp)

                # update RLS
                dx0 = x - x_prev
                da0 = a - a_prev
                dx1 = x_next - x
                self.model.update(dx0, da0, dx1)

                self._adapt_check(step, info)
            del tape
            # update state and action storage
            s_prev, a_prev, x_prev = s, a, x
            s, a, x  = s_next, a_next, x_next

            # update the multistep update variables
            c_grad_prev = c_grad
            dx1dx0_prev = dx1dx0

            # store history
            self._log(step, t, s, a, x, c)

            # break if diverged
            if np.isnan(c):
                break

            # verbose statements
            if self.verbose:
                self._info(step)
                loss_grad = (c_grad + self.gamma*lambdap_n) @ G
                print(f'eta_a_h: {self.eta_a_h}')
                print(f'loss grad: {loss_grad}')
                print(f'eps norm: {self.model.eps_norm_hist[-1]}')
                if step >1:
                    print(f'pred, actual, and epsilon dx:\n{self.model.dx_t_pred}\n{self.model.dx_t1}\n{self.model.epsilon}')
                print(f'est & true params: \n{self.model.params.T}\n{self.model.true_param}')
                print('-----------------------------------\n\n')

class IDHPnonlin():
    def __init__(self, env, config, verbose=True, seed=1) -> None:
        # TODO decide EITHER using [0] [1] or lon lat, not both
        # set seed for tensorflow
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

        self.seed  = seed
        self.gamma = config['gamma']
        self.tau   = config['tau']
        self.ms    = True if config['multistep']>0 else False

        # time at which to switch from high to low learning rates
        self.warmup_time  = config['warmup_time']
        self.error_thresh = config['error_thresh']  # error threshold in degrees (currently for alpha!)

        # adaptive switches
        self.changed         = False
        self.cooldown_1      = 0
        self.cooldown_timeit = int(config['cooldown_time']/env.dt) # 1.5 second cooldown for changing high low learning rates

        # store environment
        self.env       = env
        self.env._set_weight_matrices(config['kappa'])
        
        # setup the networks for the short period model
        self._setup_networks(config, seed)
        self.lr_decay = config['lr_decay']
        # debugging logs
        self.verbose = verbose


        self.t21, self.t22, self.t23 = 0, 0, 0
        self.t211, self.t212, self.t213, self.t214 = 0, 0, 0, 0
        self.t311, self.t312, self.t313, self.t314, self.t315 = 0, 0, 0, 0, 0
    
    def _setup_networks(self, config, seed):
        """
        Setup the IDHP agent for controlling the short period model 
        """
        ## setting up actor and critic
        # longitudinal controller
        actor_layers_lon    = config['actor_config']['layers']
        critic_layers_lon   = config['critic_config']['layers']
        in_dims_lon         = config['in_dims']
        self.hidden_dim_a_lon = list(config['actor_config']['layers'].keys())[0] # TODO, ugly, fix this
        self.hidden_dim_c_lon = list(config['critic_config']['layers'].keys())[0] # TODO, ugly, fix this

        ## creating actor critic networks
        # longitudinal controller
        self.actor         = Actor_big(in_dims_lon, actor_layers_lon, False, config['sigma'], seed, config['actor_config']['elig'])
        self.critic        = Critic_big(in_dims_lon, critic_layers_lon, False, config['sigma'], seed, config['critic_config']['elig'])
        self.target_critic = Critic_big(in_dims_lon, critic_layers_lon, False, config['sigma'], seed, config['critic_config']['elig']) # does not matter how this is initialized, it'll be reset to be equal to critic

        self.lambda_h, self.lambda_l    = config['lambda_h'], config['lambda_l'] 
        self.lambdaa = self.lambda_h
        gamma_lambda                    = self.gamma*self.lambda_h
        # TODO use list comprehension to setup the gammalambda for actor n critic...
        self.actor.gamma_lambda         = gamma_lambda
        self.critic.gamma_lambda        = gamma_lambda
        self.target_critic.gamma_lambda = gamma_lambda
        
        self.eta_a_h = config['actor_config']['eta_h']
        self.eta_c_h = config['critic_config']['eta_h']
        self.eta_a_l = config['actor_config']['eta_l']
        self.eta_c_l = config['critic_config']['eta_l']

        self.eta_a = self.eta_a_h
        
        ## setting up rls model
        # longitudinal model
        self.model = RLS(config['rls_config'])
        self.n         = self.model.state_dim
        self.m         = self.model.action_dim

        self.s_dim = config['in_dims']

    def _reset_logs(self, N):
        """
        Reset all the logs
        """
        s_dims = self.s_dim  
        e_dims = 1
        ns     = self.n 
        ms     = self.m 
        has    = self.hidden_dim_a_lon 
        hcs    = self.hidden_dim_c_lon 
        
        log = {}
        log['eta_a']  = np.zeros((N,1))
        log['t']      = np.zeros((N,1))
        log['x_full'] = np.zeros((N, 12))
        log['RSE']    = np.zeros((N,2))

        log['x']        = np.zeros((N,ns))
        log['a_cmd']    = np.zeros((N,ms))
        log['a_eff']    = np.zeros((N,ms))
        log['s']        = np.zeros((N,s_dims))
        log['yref']     = np.zeros((N,s_dims))
        log['e']        = np.zeros((N,e_dims))

        log['a_weights1'] = np.zeros((N,has*s_dims))
        log['a_weights2'] = np.zeros((N,has*ms))
        log['c_weights1'] = np.zeros((N,hcs*s_dims))
        log['c_weights2'] = np.zeros((N,hcs*ns))

        log['a_grad'] = np.zeros((N,has*s_dims + has*ms))
        log['a_elig'] = np.zeros((N,has*s_dims + has*ms))
        log['c_grad'] = np.zeros((N,hcs*s_dims + hcs*ns))
        log['c_elig'] = np.zeros((N,hcs*s_dims + hcs*ns))

        log['rls_params']   = np.zeros((N,ns*(ns+ms)))
        log['rls_cov']      = np.zeros((N,(ns+ms)**2))
        log['rls_eps_hist'] = np.zeros((N,ns))
        log['rls_eps_norm'] = np.zeros((N,1))

        self.log = log

    def _log(self, i, t, info):
        """
        Log the current algo infos
        """
        self.log['eta_a'][i] = self.eta_a
        self.log['t'][i] = t
        self.log['x_full'][i] = info['x_full']

        a_weights = self.actor.trainable_weights
        c_weights = self.critic.trainable_weights

        # for j in range(2):
        self.log['RSE'][i]      = info['RSE']
        self.log['x'][i]        = info['x'][0].flatten()
        self.log['a_cmd'][i]    = np.array(info['action_commanded'])[0].flatten()
        self.log['a_eff'][i]   = np.array(info['action_effective'])[0].flatten()
        self.log['s'][i]        = np.array(info['s'])[0].flatten()
        self.log['yref'][i]     = np.array(info['yref'])[1].flatten()
        self.log['e'][i]        = np.array(info['e'])[1].flatten()


        self.log['a_weights1'][i] = a_weights[0].numpy().flatten()
        self.log['a_weights2'][i] = a_weights[1].numpy().flatten()

        self.log['c_weights1'][i] = c_weights[0].numpy().flatten()
        self.log['c_weights2'][i] = c_weights[-1].numpy().flatten()

        self.log['rls_params'][i]   = self.model.params.flatten()
        self.log['rls_cov'][i]      = self.model.Cov.flatten()
        self.log['rls_eps_norm'][i] = self.model.eps_norm_hist[-1]
        self.log['rls_eps_hist'][i] = self.model.eps_hist[-1]

        if i > 0:
            c_grad = []
            a_grad = []
            for grad in self.critic_loss_grad:
                c_grad += list(grad.numpy().flatten())

            for grad in self.actor_loss_grad:
                a_grad += list(grad.numpy().flatten())
            
            self.log['a_grad'][i]     = np.array(a_grad)
            self.log['c_grad'][i]     = np.array(c_grad)


        # if any is nan, set every subsequent data point as nan
        if info['nans']:
            # traverse entire dictionary and set all subsequent elemnts for all arrays to be nan
            for key in self.log.keys():
                if key != 't':
                    self.log[key][i:] = np.nan
                else:
                    self.log[key][i:] = self.log[key][i-1]
            
    def _info(self, step):
        window = int(5/self.env.dt)
        window = window if step > window else step
        
        lb, ub = step-window, step
        x   = self.x_hist[lb:ub, :]
        ref = self.env.yref_hist[lb:ub]
        a   = self.a_hist[lb:ub, :]
        t   = self.t_hist[lb:ub]

        
        if step == 0:
            # plot alpha and q along with the reference in alpha
            # TODO remove hard coding for these y labels
            y_alpha = {r'$\delta \alpha$'      : [np.rad2deg(x[0,0]), 1],
                       r'$\delta \alpha_{ref}$': [np.rad2deg(ref[0]), 1]}
            y_q     = {r'$\delta q$'           : [np.rad2deg(x[0,1]), 1]}
            # TODO hard coded action scaling
            y_de    = {r'$\delta d_e$'         : [20*a, 1]}

            self.fig2, self.ax2 = make_plots(t, [y_alpha, y_q, y_de], f'States and actions', r'Time $[s]$', [r'$\delta \alpha$', r'$\delta q$', r'$\delta d_e$'], legend=True)
        
            # put window on top right
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(665,33,640,500)
        else:
            y_alpha = {r'$\delta \alpha$'      : [np.rad2deg(x[:, 0]), 1], 
                       r'$\delta \alpha_{ref}$': [np.rad2deg(ref[:]), 1]}
            y_q     = {r'$\delta q$'           : [np.rad2deg(x[:, 1]), 1]}
            y_de    = {r'$\delta d_e$'         : [20*a, 1]}             
            self.fig2, self.ax2 = update_plots(t, [y_alpha, y_q, y_de], self.fig2, self.ax2)

            plt.pause(0.02)

    def _adapt_check(self, step, info):
        """
        Adapt the learning rate
        Parameters:
        ----------
        step : int
            Current timestep
        """
        # error_threshold = np.deg2rad(self.error_thresh)                 # convert to radians
        # epsilon_norm_threshold = 5e-5                                   # rls model reset threshold on model residual
        condition_1 = step < int(self.warmup_time/self.env.dt)            # true if simulation in first time_c seconds
        # condition_2 = np.abs(info['e']) > error_threshold               # true if error is above threshold
        # condition_3 = self.model.eps_norm > epsilon_norm_threshold      # true if epsilon is above threshold
        
        # # measure the average tracking error of the last 50 timesteps
        # if step < 50:
        #     lb = 0
        # else:
        #     lb = step - 50
        # aoa_error = self.c_hist[lb:step]/self.env.kappa
        # avg_err = np.average(np.abs(aoa_error))
        # condition_4 = avg_err > 0.5      # true if average error is above 0.5 degrees

        def _decay(a, b):
            """
            a multiplied by r to approach b
            """
            c = self.lr_decay
            r = b/a
            r = c + (1-c)*r
            a = a*(0.998 + (1-0.998)*b/a)
            return np.array(a*r, np.float32)
        

        # decay cooldown for changing learning rates
        if self.cooldown_1 > 0:
            self.cooldown_1 -= 1
        # print(f"step: {step}, etac: {self.eta_c}, etaa: {self.eta_a}, cooldown: {self.cooldown_1}, condition1 {condition_1}, condition2 {condition_2}, condition3 {condition_3}")
        if not condition_1:

            if np.isclose(self.eta_a, self.eta_a_l):
                self.eta_a = self.eta_a_l
            else:
                self.eta_a = _decay(self.eta_a, self.eta_a_l)
            if np.isclose(self.eta_c, self.eta_c_l):
                self.eta_c = self.eta_c_l
            else:
                self.eta_c = _decay(self.eta_c, self.eta_c_l)
            if np.isclose(self.lambdaa, self.lambda_l):
                self.lambdaa = self.lambda_l
            else:
                self.lambdaa = _decay(self.lambdaa, self.lambda_l)
            lambda_gamma = self.lambdaa*self.gamma
        else:
            lambda_gamma = self.lambdaa*self.gamma

        current_eta_a = self.actor_optimizer.learning_rate.numpy()
        current_eta_c = self.critic_optimizer.learning_rate.numpy()
        if current_eta_a != self.eta_a and current_eta_c != self.eta_c and self.cooldown_1 <= 0:
            # print(f'\nadapting parameters at time {step*self.env.dt}s')
            # print(f'eta_a: {self.eta_a} -> {eta_a}')
            # print(f'eta_c: {self.eta_c} -> {eta_c}')
            # print(f'lambda_gamma: {self.actor.gamma_lambda} -> {lambda_gamma}')

            self.actor_optimizer.learning_rate.assign(self.eta_a)

            self.critic_optimizer.learning_rate.assign(self.eta_c)

            self.actor.gamma_lambda  = lambda_gamma
            self.critic.gamma_lambda = lambda_gamma

            # changed learning rate, so start cooldown counter
            self.cooldown_1 = self.cooldown_timeit


        # if (condition_3 and not self.changed) and not condition_1:
        #     # print(f"RLS reset, time: {step*self.env.dt}")
        #     self.model._reset()
        #     self.changed = True

    def _step_networks(self, s_prev, s_next, s, e):

        TIC = timeit.default_timer()
        with tf.GradientTape(persistent=True) as tape:
            # Get input state for networks and watch it for getting dadx
            tape.watch(s_prev)
            if self.ms:
                tape.watch(s_next)
            else:
                tape.watch(s)

            TIC1 = timeit.default_timer()
            # Get value function gradients (lambda)
            lambda_0  = self._critic(s_prev)
            lambdap_n = self._target_critic(s_next)

            TIC2 = timeit.default_timer()
            # Get input state for actor
            a_next           = self._actor(s_prev)
            action_nodes_lon = a_next[0]

            # action_nodes_lat = a_next[1]

            TIC3 = timeit.default_timer()

            F_lon, G_lon = self.model.F, self.model.G
            # F_lat, G_lat = self.model_lat.F, self.model_lat.G

        # self.actor.update_traces(test_jacobian)
        TOC = timeit.default_timer()

        grad_lon = tape.gradient(action_nodes_lon, s_prev)
        
        TUNC = timeit.default_timer()

        dadx_lon   = tf.reshape(grad_lon[:self.n], (1, self.n)) 
        dx1dx0_lon = F_lon + G_lon @ dadx_lon # TODO check theoretiaclly what this should look like...
        dx1dx0_lon = dx1dx0_lon.numpy()

        G_lat, dx1dx0_lat = None, None

        lambdas = [lambda_0, lambdap_n]
        Gs      = [G_lon, G_lat]
        dx1dx0s = [dx1dx0_lon, dx1dx0_lat]

        TONC = timeit.default_timer()

        # profiling times
        self.t21 += TOC - TIC
        self.t22 += TUNC - TOC
        self.t23 += TONC - TUNC

        self.t211 += TIC1 - TIC
        self.t212 += TIC2 - TIC1
        self.t213 += TIC3 - TIC2
        self.t214 += TOC - TIC3
        return a_next, lambdas, Gs, dx1dx0s, tape
    
    def _update_networks(self, lambdas, dx1dx0s, dx1dx0_prevs, c_grad, c_grad_prev, Gs, s_prev, a_next, a, tape):
        
        tic1 = timeit.default_timer()
        dx1dx0_prev_lon, _  = dx1dx0_prevs
        dx1dx0_lon, _       = dx1dx0s
        lambda_0, lambdap_n = lambdas
        G_lon, _            = Gs

        # update critic
        if self.ms:
            td_error = lambda_0[0] - c_grad_prev[0] - self.gamma*c_grad[0] @ dx1dx0_prev_lon - (self.gamma**2)*lambdap_n[0] @ dx1dx0_lon @ dx1dx0_prev_lon
        else:
            td_error = lambda_0[0] - c_grad[0] - self.gamma*lambdap_n[0] @ dx1dx0_lon
        
        tic2 = timeit.default_timer()
        self.critic_loss_grad = tape.gradient(lambda_0[0], self.critic.trainable_weights, output_gradients=td_error)

        tic3 = timeit.default_timer()
        self.critic_optimizer.apply_gradients(zip(self.critic_loss_grad, self.critic.trainable_weights))

        # update target critic
        self.target_critic.soft_update(self.critic.trainable_weights, tau=self.tau)
        tic4 = timeit.default_timer()

        # update actor
        s_random = tf.random.normal(shape=(1,1),
                                    mean=s_prev,
                                    stddev=[0.010,0.010,0.008,0.003])
        a_random = self.actor(s_random, trace=True)
        L_T = tf.sqrt((a[0]-a_next[0])**2)
        L_S = tf.sqrt((a[0]-a_random[0])**2)

        # self.lambda_t, self.lambda_s = 0.0, 0.00
        self.lambda_t, self.lambda_s = 0.012, 0.001

        loss_grad = -(c_grad[0] + self.gamma*lambdap_n[0]) @ G_lon + self.lambda_t*L_T + self.lambda_s*L_S # c_grad techinically shouldnt exists, but with smoothness assumption this works
        # self.actor_loss_grad = tape.gradient(a_next[0], self.actor.trainable_weights, output_gradients=loss_grad)
        self.actor_loss_grad = self.actor.get_weight_update(loss_grad)
        self.actor_optimizer.apply_gradients(zip(self.actor_loss_grad, self.actor.trainable_weights))

        tic5 = timeit.default_timer()
        del tape

        tic6 = timeit.default_timer()

        self.t311 += tic2 - tic1
        self.t312 += tic3 - tic2
        self.t313 += tic4 - tic3
        self.t314 += tic5 - tic4
        self.t315 += tic6 - tic5
        
    def _setup_optimizers(self):
        """
        Setup the optimizers for idhp using the short period model
        """
        self.eta_a = self.eta_a_h
        self.eta_c = self.eta_c_h

        self.actor_optimizer  = tf.keras.optimizers.SGD(learning_rate=self.eta_a)
        self.critic_optimizer = tf.keras.optimizers.SGD(learning_rate=self.eta_c)
    
    def _actor(self, s):
        """
        Get the actor output
        Parameters:
        ----------
        s : np.array
            The state of the MDP
        Returns:
        -------
        np.array
            The action to be taken, de, da, dr
        """
        s_lon = tf.reshape(s,(1,self.s_dim))
        a_lon = self.actor(s_lon)
        a_lat = None
        return a_lon, a_lat
    
    def _critic(self, s):
        """
        Get the critic output
        """
        s_lon = tf.reshape(s,(1,self.s_dim))
        c_lon = self.critic(s_lon)
        c_lat = None

        return [c_lon, c_lat]
     
    def _target_critic(self, s):
        """
        Get the target critic output
        """
        s_lon = tf.reshape(s,(1,self.s_dim))
        c_lon = self.target_critic(s_lon)
        c_lat = None

        return [c_lon, c_lat]
    
    def _get_action(self, a):
        """
        Get the action from the actor network
        """
        de = a[0].numpy()[0,0]
        da, dr = 0, 0

        return np.array([de, da, dr])
    
    def train(self):
        """
        Train the IDHP agent using multi step method
        """
        t1, t2, t3, t4 = 0, 0, 0, 0 # profiling times
        t31, t32 = 0, 0

        ## Initialize algorithm

        # initialize target critic to be identical to critic
        self.target_critic.soft_update(self.critic.trainable_weights, tau=1)
        # self.target_critic_lat.soft_update(self.critic_lat.trainable_weights, tau=1)
        
        # initialize all the loop variables
        t, dt, t_end        = 0, self.env.dt, self.env.t_end
        steps               = int(t_end/dt)
        s, c, done, _, info = self.env.reset(seed=self.seed)
        s                   = tf.convert_to_tensor(s, dtype=tf.float32)
        a                   = self._actor(s)
        x_lon               = info['x'][0].copy()
        s_prev, a_prev      = s, None
        x_prev_lon          = None

        # multistep update variables
        c_grad, c_grad_prev = info['reward_grad'].copy(), None
        dx1dx0_prev_lon     = None
        dx1dx0_prev_lat     = None

        self._reset_logs(steps)
        self._setup_optimizers()
        
        self.RSE     = [0, 0]
        
        ## Start of algorithm
        for step in (bar := tqdm(range(steps))):
            tic = timeit.default_timer()
            bar.set_description(f"    t, RSE = {t:.2f} s, {np.rad2deg(info['RSE'][0]):.2e} deg")
            
            t = dt*(step+1)
            # execute action and observe new state, 
            s_next, c, done, _, info = self.env.step(self._get_action(a))
            x_next_lon = info['x'][0].copy()
            s_next = tf.convert_to_tensor(s_next, dtype=tf.float32)
            e      = info['e'].copy()
            c_grad = info['reward_grad'].copy()
            # cumulating RSE score
            self.RSE[0] += info['RSE'][0]
            self.RSE[1] += info['RSE'][1]

            toc = timeit.default_timer()
            # a_next = s_next*K_p
            # a_next = (tf.reshape(a_next[0], (1,1)), tf.reshape(a_next[1:], (1,2)))
            a_next, lambdas, Gs, dx1dx0s, tape = self._step_networks(s_prev, s_next, s, e)

            tunc = timeit.default_timer()
            # RLS update rule requires 2 past timestep's states and actions, for convenience RLS & networks 
            # are updated together after 0th and 1st timestep (0th time step taken before loop starts)
            if step > 0:
                dx1dx0_prevs = [dx1dx0_prev_lon, dx1dx0_prev_lat]
                # update networks
                self._update_networks(lambdas, dx1dx0s, dx1dx0_prevs, c_grad, c_grad_prev, Gs, s_prev, a_next, a, tape)

                tic1 = timeit.default_timer()
                # update RLS
                dx0_lon = x_lon - x_prev_lon
                da0_lon = (a[0] - a_prev[0]).numpy().T
                dx1_lon = x_next_lon - x_lon
                self.model.update(dx0_lon, da0_lon, dx1_lon)

                self._adapt_check(step, info)
            
            tomc = timeit.default_timer()
            # update state and action storage
            s_prev, a_prev = s, a
            s, a           = s_next, a_next
            x_prev_lon     = x_lon
            x_lon          = x_next_lon

            dx1dx0_lon, dx1dx0_lat = dx1dx0s
            # update the multistep update variables
            c_grad_prev     = c_grad
            dx1dx0_prev_lon = dx1dx0_lon

            # store history
            self._log(step, t, info)

            tenc = timeit.default_timer()

            t1 += toc-tic
            t2 += tunc-toc
            t3 += tomc-tunc
            t4 += tenc-tomc
            
            try:
                t31 += tic1-tunc
                t32 += tomc-tic1
            except:
                pass

            # break if diverged
            if info['nans']:
                break
            
        # profiling times
        if self.verbose:
            print(f"Profling times:\nt1: {t1:.3f}\nt2: {t2:.3f} (t21: {self.t21:3f}, t22: {self.t22:3f}, t23: {self.t23:3f})\nt3: {t3:.3f} (t31: {t31:.3f}, t32: {t32:.3f})\nt4: {t4:.3f}")
            print(f'\nt211: {self.t211:.3f}, t212: {self.t212:.3f}, t213: {self.t213:.3f}, t214: {self.t214:.3f}')
            print(f't311: {self.t311:.3f}, t312: {self.t312:.3f}, t313: {self.t313:.3f}, t314: {self.t314:.3f}, t315: {self.t315:.3f}')


if __name__ == "__main__":
    test_network = Critic_big(4, {10: 'tanh', 3: 'tanh'}, False)
    a = 1

    s = tf.constant([[1.0, 1.0, 12.0, 1.3]], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        # Get input state for networks and watch it for getting dadx
        tape.watch(s)
        out = test_network(s)
        a= 12

    hehe = 123