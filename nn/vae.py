#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class VAE:
    def __init__(self, input_dim, enc_hid_dim, n_enc_layer, latent_dim, dec_hid_dim, n_dec_layer,
                 init_lr, n_sample, beta, use_batch_norm, **kargs):
        self.input_dim = input_dim

        self.enc_hid_dim = enc_hid_dim
        self.n_enc_layer = n_enc_layer

        self.latent_dim = latent_dim

        self.dec_hid_dim = dec_hid_dim
        self.n_dec_layer = n_dec_layer

        self.init_lr = init_lr
        self.n_sample = n_sample # not use yet

        self.beta = beta
        self.use_batch_norm = use_batch_norm

    def batch_normalize(self, data, scope, phase):

        norm_data = tf.contrib.layers.batch_norm(
            data,
            decay=0.9,
            center=True,
            scale=True,
            is_training=phase,
            scope=scope)

        return norm_data

    def __build_layer(self, input, input_dim, output_dim, act_func = lambda x: x, name_scope = 'vae_'):
        with tf.variable_scope(name_scope):
            W = tf.get_variable(name='W',shape=[input_dim,output_dim],dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',shape=[output_dim],dtype=tf.float32,initializer=tf.zeros_initializer())

            a = act_func(tf.nn.xw_plus_b(input,W,b,name='mul_W_add_b'))

            return a

    def __build_encoder(self, input, input_dim, enc_hid_dim, n_enc_layer, latent_dim):
        assert n_enc_layer > 0

        with tf.variable_scope('encoder'):
            a = None

            # build multi layers
            for i_layer in range(n_enc_layer):
                if i_layer == 0:
                    a = self.__build_layer(input=input,input_dim=input_dim,output_dim=enc_hid_dim,act_func=tf.nn.elu,
                                           name_scope='encoder_layer_%d' % i_layer)
                    # batch normalization
                    if self.use_batch_norm is True:
                        a = self.batch_normalize(data=a,scope='encoder_layer_bn_%d' % i_layer,phase=self.phase)
                else:
                    a = self.__build_layer(input=a, input_dim=enc_hid_dim, output_dim=enc_hid_dim, act_func=tf.nn.elu,
                                           name_scope='encoder_layer_%d' % i_layer)

            # build encoding vector
            mu  = self.__build_layer(input=a,input_dim=enc_hid_dim,output_dim=latent_dim,name_scope='encoder_mu')
            var = self.__build_layer(input=a, input_dim=enc_hid_dim,output_dim=latent_dim,name_scope='encoder_var')

        return mu, var

    def __sample(self, mu, var):
        return mu + var * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    def __build_decoder(self, input, input_dim, latent_dim, dec_hid_dim, n_dec_layer):
        assert n_dec_layer > 0

        with tf.variable_scope('decoder'):
            a = None

            # build multi layers
            for i_layer in range(n_dec_layer):
                if i_layer == 0:
                    a = self.__build_layer(input=input, input_dim=input_dim,output_dim=dec_hid_dim,
                                           act_func=tf.nn.elu, name_scope='decoder_layer_%d' % i_layer)
                    # batch normalization
                    if self.use_batch_norm:
                        a = self.batch_normalize(data=a, scope='decoder_layer_bn_%d' % i_layer, phase=self.phase)
                else:
                    a = self.__build_layer(input=a, input_dim=dec_hid_dim, output_dim=dec_hid_dim, act_func=tf.nn.elu,
                                           name_scope='decoder_layer_%d' % i_layer)

            y = self.__build_layer(input=a,input_dim=dec_hid_dim,output_dim=latent_dim,name_scope='decoder_y')

        return y

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

    def __build_placeholder(self, input_dim, init_lr):
        input = tf.placeholder(name='input',dtype=tf.float32,shape=[None,input_dim])
        lr = tf.placeholder_with_default(input=init_lr,shape=[],name='lr_with_default')
        phase = tf.placeholder_with_default(input=False,shape=[],name='phase_with_default')

        return input, lr, phase

    def build_normalize(self, train_data):
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data)

    def transform_raw_data(self, raw_data):
        return self.scaler.transform(raw_data)

    def inverse_transform_raw_data(self, norm_data):
        return self.scaler.inverse_transform(norm_data)

    def batch_train(self, batch_data, lr):
        feed_dict = {
            self.input: batch_data,
            self.lr: lr,
            self.phase: 1
        }

        loss, _ = self.sess.run([self.vae_loss,self.train_step],feed_dict)

        return loss

    def get_decoded_output(self, datas):
        feed_dict = {
            self.input: datas
        }

        decoded_output = self.sess.run(self.decoded_output, feed_dict=feed_dict)

        return decoded_output

    def save(self, save_path):
        self.saver.save(self.sess, save_path)

    def restore(self, restore_path):
        self.saver.restore(self.sess, restore_path)

    def build(self):
        # build necessary placeholders
        self.input, self.lr, self.phase = self.__build_placeholder(input_dim=self.input_dim, init_lr=self.init_lr)

        # encoder == approximate q(z|X)
        self.mu, self.var = self.__build_encoder(input=self.input,input_dim=self.input_dim,enc_hid_dim=self.enc_hid_dim,
                                                     n_enc_layer=self.n_enc_layer,latent_dim=self.latent_dim)



        # sampling
        self.zs = self.__sample(mu=self.mu,var=self.var)

        # decoder == approximate p(X|z)
        self.hat_input = self.__build_decoder(input=self.zs,input_dim=self.latent_dim,latent_dim=self.input_dim,
                                              dec_hid_dim=self.dec_hid_dim, n_dec_layer=self.n_dec_layer)

        self.decoded_output = tf.nn.sigmoid(self.hat_input)

        # loss
        self.recons_loss, self.kl_loss, self.vae_loss = self.__build_loss(input=self.input,hat_input=self.hat_input,
                                                                                  mu=self.mu,var=self.var)

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

    input_dim = 100
    enc_hid_dim = 200
    n_enc_layer = 2
    latent_dim = 100
    dec_hid_dim = 200
    n_dec_layer = 2
    init_lr = 0.001
    n_sample = 2
    beta = 0.0
    use_batch_norm = False

    vae = VAE(input_dim=input_dim, enc_hid_dim=enc_hid_dim, n_enc_layer=n_enc_layer, latent_dim=latent_dim,
              dec_hid_dim=dec_hid_dim, n_dec_layer=n_dec_layer, init_lr=init_lr, n_sample=n_sample, beta=beta,
              use_batch_norm=use_batch_norm)

    vae.build()

