import tensorflow as tf
import numpy as np
import os ,warnings
# Utility Functions for neuronal networks in general, variational autoencoders and e2c


# initialization functions:
# Xavier Glorot and Yoshua Bengio(2010):
# Understanding the difficulty of training deep feedforward neural networks.
# International conference on artificial intelligence and statistics.
# "border" was chosen according to paper
def glorot_uniform_initializer(shape, dtype=tf.float32, constant=1, seed=None):
    n_in = shape[0]
    n_out = shape[1]
    boarder = constant * np.sqrt(6.0/(n_in + n_out))
    return tf.random_uniform((n_in, n_out), minval=-boarder, maxval=boarder, dtype=tf.float32, seed=None)


# Andrew Saxe, James L. McClelland, Surya Ganguli
# Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.
# arXiv preprint arXiv:1312.6120 (2013).
# Code from Lasagne https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def orthogonal_init(shape, dtype=tf.float32):
    gain = tf.sqrt(2.0)
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                           "supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = tf.random_normal(flat_shape, 0.0, 1.0, dtype=tf.float32)
    u, _, v = np.linalg.svd(a.eval(), full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q


# Init weights for dense layer
def init_wight_and_bias(input_dim, output_dim, initialization=glorot_uniform_initializer,
                        name_weight=None, name_bias=None):
    w = tf.Variable(initialization((input_dim, output_dim)), name=name_weight)
    b = tf.Variable(tf.zeros([output_dim]), name=name_bias)
    return w, b


# Dummy Activation Function
def identity_activation(x):
    return x

# Re parameterization Trick for normal distribution. See Kingma, Welling: Auto-Encoding Variational Bayes
def reparameterize_normal(mean, sig, training, diag_only=True):
    eps = tf.random_normal(tf.shape(mean), mean=0, stddev=1)
    if diag_only:
        reparameterized = mean + sig * eps
    else:
        reparameterized = mean + tf.squeeze(tf.batch_matmul(sig, tf.expand_dims(eps, 2)))
    return tf.cond(training, lambda: reparameterized, lambda: mean)

# Cost functions
def binary_crossentropy(true, pred, epsilon=1e-8):
    return - tf.reduce_mean(tf.reduce_sum(true * tf.log(pred + epsilon) + (1 - true) * tf.log(1 - pred + epsilon), 1))

def mse(true, pred):
    return tf.reduce_mean(tf.square(true - pred))

def se(true, pred):
    return tf.reduce_mean(tf.reduce_sum(tf.square(true - pred), 1))

#def gaussian_kl_N01Prior(mean, log_sig):
#    return - tf.reduce_mean(tf.reduce_sum(0.5 * (1 + 2 * log_sig - mean ** 2 - tf.exp(2*log_sig)), 1))

#def gaussian_kl(mean_0, log_sig_0, mean_1, log_sig_1):
#    first_term = log_sig_1 - log_sig_0
#    second_term = tf.exp(2 * log_sig_0) + (mean_0 - mean_1)**2
#    third_term = 2 * tf.exp(2 * log_sig_1)
#    loss = tf.reduce_sum(-0.5 + first_term + second_term/third_term, 1)
#    loss_avg = tf.reduce_mean(loss)
#    return loss_avg

def full_gaussian_kl(mean_1, sig_1, mean_2=None, sig_2=None):
    if mean_2 is None:
        mean_2 = tf.zeros(shape=tf.shape(mean_1))
    if sig_2 is None:
        sig_2 = tf.matrix_diag(tf.ones(shape=tf.shape(mean_1)))
    dist_1 = tf.contrib.distributions.MultivariateNormalFull(loc=mean_1, sigma=sig_1)
    dist_2 = tf.contrib.distributions.MultivariateNormalFull(loc=mean_2, sigma=sig_2)
    kl = tf.contrib.distributions.kl(dist_1, dist_2)
    return tf.reduce_mean(kl)

def gaussian_kl(mean_1, sig_1, mean_2=None, sig_2=None):
    if mean_2 is None:
        mean_2 = tf.zeros(shape=tf.shape(mean_1))
    if sig_2 is None:
        sig_2 = tf.ones(shape=tf.shape(sig_1))
    dist_1 = tf.contrib.distributions.MultivariateNormalDiag(loc=mean_1, scale_diag=sig_1)
    dist_2 = tf.contrib.distributions.MultivariateNormalDiag(loc=mean_2, scale_diag=sig_2)
    kl = dist_1.kl_divergence(dist_2)
    return tf.reduce_mean(kl)

#copyed from http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
#def gau_kl(pm, pv, qm, qv):
#    """
#    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#    Also computes KL divergence from a single Gaussian pm,pv to a set
#    of Gaussians qm,qv.
#    Diagonal covariances are assumed.  Divergence is expressed in nats.
#    """
#    # Determinants of diagonal covariances pv, qv
#    dpv = tf.reshape(tf.reduce_prod(pv, 1), [1])
#    dqv = tf.reshape(tf.reduce_prod(qv, 1), [1])
#    # Inverse of diagonal covariance qv
#    iqv = 1./qv
#    # Difference between means pm, qm
#    diff = qm - pm
#    n = tf.cast(tf.shape(pm)[1], tf.float32)
#    return tf.reduce_mean(0.5 * (tf.log(dqv / dpv)               # log |\Sigma_q| / |\Sigma_p|
#             + tf.reduce_sum(iqv * pv, 1)          # + tr(\Sigma_q^{-1} * \Sigma_p)
#             + tf.reduce_sum(diff * iqv * diff, 1) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
#             - n))                     # - N


#def _calc_normal_normal_kld(mu_0, mu_1, sig_sq_0, sig_sq_1):
#    inv_sig_sq_1 = tf.inv(sig_sq_1)
#    tr1 = tf.mul(inv_sig_sq_1, sig_sq_0)
#    tr_term = tf.reduce_sum(tr1, 1)

#    diff = tf.square(tf.sub(mu_1, mu_0))
#    diff_term = tf.reduce_sum(tf.mul(diff, inv_sig_sq_1))

#    det_1 = tf.reduce_prod(sig_sq_1, 1)
#    det_0 = tf.reduce_prod(sig_sq_0, 1)
#    det_term = tf.log(tf.div(det_1, det_0))
#    return 0.5 * tf.reduce_mean(diff_term + tr_term + det_term - 3)

def normalize_euclidean(x):
    norm_fact = tf.expand_dims(tf.rsqrt(tf.reduce_sum(tf.square(x), 1)), 1)
    return x * norm_fact

def euclidean_norm_factor(x):
    return tf.expand_dims(tf.rsqrt(tf.reduce_sum(tf.square(x), 1)), 0)

def binary_kl(rho, rho_hat):
    return tf.reduce_sum(_logfunc(rho, rho_hat) + _logfunc(1 - rho, 1 - rho_hat), 0)

def _logfunc(x1, x2):
    x1 = tf.maximum(x1, 1e-8)
    x2 = tf.maximum(x2, 1e-8)
    return x1 * tf.log(tf.div(x1, x2))

class DenseLayer():

    def __init__(self,
                 input_dim,
                 output_dim,
                 name=None,
                 activation=tf.nn.relu,
                 initializer=glorot_uniform_initializer,
                 seed=None):

        self.name = name
        self.w = tf.get_variable(name+'_weight', shape=[input_dim, output_dim], initializer=initializer)
        self.b = tf.get_variable(name+'_bias', shape=[output_dim], initializer=tf.constant_initializer(0))
        self.activation = activation

    def forward_prop(self, input):
        return self.activation(tf.matmul(input, self.w) + self.b)

    def get_histogram_logs(self, prefix=''):
        tf.histogram_summary(prefix + self.name + '_weight', self.w)
        tf.histogram_summary(prefix + self.name + '_bias', self.b)

    def get_vars(self):
        return [self.w, self.b]


class PathCreatingSaver(tf.train.Saver):

    def __init__(self, folder_path, var_list=None, reshape=False, sharded=False, max_to_keep=5,
                 keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, saver_def=None,
                 builder=None):
        super().__init__(var_list=var_list, reshape=reshape, sharded=sharded, max_to_keep=max_to_keep,
                         keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, name=name,
                         restore_sequentially=restore_sequentially, saver_def=saver_def, builder=builder)
        if not os.path.exists(folder_path):
            warnings.warn('Path ' +folder_path+' not found - creating')
            os.makedirs(folder_path)


