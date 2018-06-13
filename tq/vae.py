import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import utils

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""

    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))

    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, tranfer_fct= tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.tranfer_fct = tranfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        #tf graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        #create autoencoder network
        self._create_network()

        #loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()

        #initializing the tensor flow variables
        init = tf.global_variables_initializer()

        #launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        #initialize autoencode network weights and bias
        network_weights = self._initialize_weights(**self.network_architecture)

        #use recognition network to determine mean and (log) variance of Gaussian distribuion in latent space
        self.z_mean, self.z_log_sigma_sq= self._recognition_network(network_weights["weights_recog"],
                                                                    network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        # n_z = self.network_architecture["n_z"]
        eps = tf.random_normal(tf.shape(self.z_mean), 0, 1, dtype=tf.float32)

        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # use generator to determine mean of Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._generator_network(network_weights["weights_gener"],
                                                       network_weights["biases_gener"])
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1, n_hidden_gener_2,n_input,n_z):
        all_weights = dict()
        all_weights['weights_recog']={
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1,n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))
        }

        all_weights['biases_recog']={
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }

        all_weights['weights_gener']={
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))
        }

        all_weights['biases_gener']={
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }

        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which maps inputs onto a normal distribution in latent space. The transformation is parametrized and can be learned
        layer_1 = self.tranfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.tranfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decode (decode network), which maps points in latent space onto a Bernoulli distribution in data space
        # The transformation is parametrized and can be learned
        layer_1 = self.tranfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.tranfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))

        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of tow terms :
        # 1.) The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli distribution
        #     inducted by the decoder in the data space)
        #     This can be interpreted as the number of "nats" required for reconstructing the input when the activation in latent is given.
        # Additing 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = - tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) +
                                     (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)

        # 2.) The latent loss, which is defined as the Kullback Leibler (KL) divergence between the distribution in latent space induced by the encoder on
        # the data and some prior. This acts as a kind of regularizer. This  can be interpreted as the number of "nats" required for transmitting the latent space distribution given the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                            - tf.square(self.z_mean)
                                            - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss) #average over batch

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, x):
        """Train model based on mini-batch of input data
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x : x})

        return cost

    def transform(self, x):
        """Transform data by mapping it into the latent space."""
        # Note : this maps to mean of distribution, we could afternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x : x})

    def generate(self, z_mu=None):
        """Generate data by mapping from latent space.

        If z_mu is not none, data for this point in latent space is generated. Otherwise, z_mu is
        drawn from prior in latent space.
        """

        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])

        # Note: this maps to mean of distribution, we could alternatively sample from
        # Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})

    def reconstruct(self, x):
        """ Use VAE to reconstruct given data."""
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x : x})

    def build_normalize(self, train_data):
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data)

    def transform_raw_data(self, raw_data):
        return self.scaler.transform(raw_data)