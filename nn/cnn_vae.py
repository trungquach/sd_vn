#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class CNN_VAE:
    def __init__(self, input_dim, init_lr, beta, use_batch_norm, filter_sizes, n_filters, **kargs):
        self.input_dim = input_dim
        self.init_lr = init_lr
        self.beta = beta
        self.use_batch_norm = use_batch_norm
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters

    def __build_placeholder(self, input_dim, init_lr):
        input = tf.placeholder(name='input',dtype=tf.float32,shape=[None,input_dim])
        lr = tf.placeholder_with_default(input=init_lr,shape=[],name='lr_with_default')
        phase = tf.placeholder_with_default(input=False,shape=[],name='phase_with_default')

        return input, lr, phase

    def __build_layer(self, input, input_dim, output_dim, act_func = lambda x: x, name_scope = 'vae_'):
        with tf.variable_scope(name_scope):
            W = tf.get_variable(name='W',shape=[input_dim,output_dim],dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',shape=[output_dim],dtype=tf.float32,initializer=tf.zeros_initializer())

            a = act_func(tf.nn.xw_plus_b(input,W,b,name='mul_W_add_b'))

            return a

    def __build_encoder(self, input, input_dim, filter_sizes, n_filters, **kargs):
        input_4d = tf.reshape(input,shape=[-1,1,input_dim,1])
        hid_layer_shapes = [input_4d.get_shape().as_list()]
        hid_weigh_shapes = []

        previous_input = input_4d
        with tf.variable_scope('encoder'):
            for id, (filter_size, n_filter) in enumerate(zip(filter_sizes, n_filters)):
                with tf.variable_scope('encoder_conv_%i' % id):
                    in_channel = hid_layer_shapes[-1][-1]

                    W = tf.get_variable(name='W',shape=[1,filter_size,in_channel,n_filter],dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable(name='b',shape=[n_filter],dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

                    conv = tf.nn.conv2d(
                        previous_input,
                        W,
                        strides=[1,1,2,1],
                        padding='SAME',
                        name='conv'
                    )

                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                    # save previous
                    previous_input = h
                    hid_layer_shapes.append(previous_input.get_shape().as_list())
                    hid_weigh_shapes.append(W.get_shape().as_list())

            out_flatten = tf.contrib.layers.flatten(h)

            in_dim = out_flatten.get_shape().as_list()[-1]
            mu = self.__build_layer(input=out_flatten, input_dim=in_dim, output_dim=in_dim, name_scope='encoder_mu')
            var = self.__build_layer(input=out_flatten, input_dim=in_dim, output_dim=in_dim, name_scope='encoder_var')

        del hid_layer_shapes[-1]
        return mu, var, in_dim, hid_layer_shapes, hid_weigh_shapes

    def __build_decoder(self, input, input_dim, hid_layer_shapes, hid_weigh_shapes):
        input_4d = tf.reshape(input,shape=[-1,1,input_dim,1])

        previous_input = input_4d
        with tf.variable_scope('decoder'):
            for id, (hid_layer_shape, hid_weigh_shape) in enumerate(zip(hid_layer_shapes, hid_weigh_shapes)):
                with tf.variable_scope('decoder_deconv_%i' %id):
                    W = tf.get_variable(name='W', shape=hid_weigh_shape, dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable(name='b', shape=hid_layer_shape[-1], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

                    deconv = tf.nn.conv2d_transpose(
                        previous_input,
                        W,
                        output_shape=[tf.shape(input)[0], hid_layer_shape[1], hid_layer_shape[2], hid_layer_shape[3]],
                        strides=[1, 1, 2, 1],
                        padding='SAME',
                        name='deconv'
                    )

                    h = tf.nn.relu(tf.nn.bias_add(deconv, b), name="relu")

                    # save previous
                    previous_input = h

        return tf.contrib.layers.flatten(previous_input)

    def __sample(self, mu, var):
        return mu + var * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    def __build_loss(self, input, hat_input, mu, var):
        # reconstruction loss

        recons_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=hat_input, labels=input)
        recons_loss = tf.reduce_sum(recons_loss,axis=1)

        # kl
        eps = 1e-10
        kl_loss =  0.5 * tf.reduce_sum(tf.square(var) + tf.square(mu)
                                       - 1 - tf.log(eps + tf.square(var)),axis=1)

        vae_loss = tf.reduce_mean(recons_loss + self.beta * kl_loss)

        return recons_loss, kl_loss, vae_loss

    def build(self):
        # build necessary placeholders
        self.input, self.lr, self.phase = self.__build_placeholder(input_dim=self.input_dim, init_lr=self.init_lr)

        # encoder == approximate q(z|X)
        self.mu, self.var, self.latent_dim, hid_layer_shapes, hid_weigh_shapes = self.__build_encoder(input=self.input,
                                    input_dim=self.input_dim,filter_sizes=self.filter_sizes,n_filters=self.n_filters)

        # sampling
        self.zs = self.__sample(mu=self.mu,var=self.var)

        # decoder == approximate p(X|z)
        hid_layer_shapes.reverse()
        hid_weigh_shapes.reverse()

        self.hat_input = self.__build_decoder(input=self.zs, input_dim=self.latent_dim, hid_layer_shapes=hid_layer_shapes,
                                              hid_weigh_shapes=hid_weigh_shapes)

        self.decoded_output = tf.nn.sigmoid(self.hat_input)

        # loss
        self.recons_loss, self.kl_loss, self.vae_loss = self.__build_loss(input=self.input, hat_input=self.hat_input,
                                                                          mu=self.mu, var=self.var)

        # optimizer
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.vae_loss)

        # session
        self.saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

if __name__ == '__main__':
    # test
    '''
    self, input_dim, enc_hid_dim, n_enc_layer, latent_dim, dec_hid_dim, n_dec_layer,
    init_lr, n_sample, ** kargs
    '''

    input_dim = 1024
    enc_hid_dim = 200
    n_enc_layer = 2
    latent_dim = 100
    dec_hid_dim = 200
    n_dec_layer = 2
    init_lr = 0.001
    n_sample = 2
    beta = 0.0
    use_batch_norm = False
    filter_sizes = [11,5,3]
    n_filters = [5,5,1]

    vae = CNN_VAE(input_dim=input_dim, enc_hid_dim=enc_hid_dim, n_enc_layer=n_enc_layer, latent_dim=latent_dim,
              dec_hid_dim=dec_hid_dim, n_dec_layer=n_dec_layer, init_lr=init_lr, n_sample=n_sample, beta=beta,
              use_batch_norm=use_batch_norm, filter_sizes=filter_sizes, n_filters=n_filters)

    vae.build()

