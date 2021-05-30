import self as self
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from layers import conv_layer
from config import MSG_LEN,BATCH_SIZE,NUM_EPOCHS, LEARNING_RATE
from utils import init_weights, gen_data


matplotlib.use('TkAgg')

class CryptoNet(object):
    def __init__(self, sess, msg_len=MSG_LEN, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        """

        :param sess: TensorFlow session
        :param msg_len:The length of the input message to encrypt
        :param batch_size:Length of Chatur and deepayni's private key
        :param epochs:Minibatch size for each adversarial training
        :param learning_rate:Learning Rate for Adam Optimizer
        """
        self.sess = sess
        self.msg_len = msg_len
        self.key_len = self.msg_len
        self.N = self.msg_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.build_model()

    # Weights for fully connected layers
    def build_model(self):
            self.w_Sender = init_weights("Sender_w", [2 * self.N, 2 * self.N])
            self.w_Reciever = init_weights("Reciever_w", [2 * self.N, 2 * self.N])
            self.w_listener1 = init_weights("listener_w1", [self.N, 2 * self.N])
            self.w_listener2 = init_weights("listener_w2", [2 * self.N, 2 * self.N])






            self.msg = tf.placeholder("float", [None, self.msg_len])
            self.key = tf.placeholder("float", [None, self.key_len])

    # Sender's network
    # FC layer -> Conv Layer (4 1-D convolutions)

            self.Sender_input = tf.concat([self.msg, self.key], 1)# a 4*1 matrix
            self.Sender_hidden = tf.nn.sigmoid(tf.matmul(self.Sender_input, self.w_Sender))
            self.Sender_hidden = tf.expand_dims(self.Sender_hidden, 2)
            self.Sender_output = tf.squeeze(conv_layer(self.Sender_hidden, "Sender"))

    # reciever's network
    # FC layer -> Conv Layer (4 1-D convolutions)

            self.Reciever_input = tf.concat([self.Sender_output, self.key], 1)#she will have chatur's ciphertext
            self.Reciever_hidden = tf.nn.sigmoid(tf.matmul(self.Reciever_input, self.w_Reciever))
            self.Reciever_hidden = tf.expand_dims(self.Reciever_hidden, 2)
            self.Reciever_output = tf.squeeze(conv_layer(self.Reciever_hidden, "Reciever"))

    # listener's network(he is the listener who will try to make a guess)
    # FC layer -> FC layer -> Conv Layer (4 1-D convolutions)
            self.listener_input = self.Sender_output# evesdropping ciphertext
            self.listener_hidden1 = tf.nn.sigmoid(tf.matmul(self.listener_input, self.w_listener1))
            self.listener_hidden2 = tf.nn.sigmoid(tf.matmul(self.listener_hidden1, self.w_listener2))
            self.listener_hidden2 = tf.expand_dims(self.listener_hidden2, 2)
            self.listener_output = tf.squeeze(conv_layer(self.listener_hidden2, "listener"))


#Loss Function
    def train(self):
        self.decrypt_err_listener = tf.reduce_mean(tf.abs(self.msg - self.listener_output))
        self.decrypt_err_reciver = tf.reduce_mean(tf.abs(self.msg - self.Reciever_output))
        self.loss_reciever = self.decrypt_err_reciver + (1. - self.decrypt_err_listener) ** 2


#Get training variables corresponding to each network
        self.t_vars = tf.trainable_variables()
        self.Chatur_or_deepayni_vars = [var for var in self.t_vars if 'Sender_' in var.name or 'Reciever_' in var.name]
        self.prakash_vars=  [var for var in self.t_vars if 'listener_' in var.name]

# Build the optimizers

        self.deepayni_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_reciever, var_list=self.Chatur_or_deepayni_vars)
        self.prakash_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.decrypt_err_listener, var_list=self.prakash_vars)

        self.deepayni_errors, self.prakash_errors = [], []

# Begin Training
        tf.global_variables_initializer().run()
        for i in range(self.epochs):
            iterations = 2000
            print('Training Sender and reciever, Epoch:', i + 1)
            deepayni_loss, _ = self._train('Reciever', iterations)
            self.deepayni_errors.append(deepayni_loss)
            print('Training listener, Epoch:', i + 1)
            _, prakash_loss = self._train('listener', iterations)
            self.prakash_errors.append(prakash_loss)

        self.plot_errors()

    def _train(self, network, iterations):
        global decrypt_err
        deepayni_decrypt_error, prakash_decrypt_error = 1., 1.
        bs = self.batch_size

        if network == 'listener':
            bs *= 2
        for i in range(iterations):
             msg_in_val, key_val = gen_data(n=bs, msg_len=self.msg_len, key_len=self.key_len)


        if network == 'Reciever':

           _, decrypt_err = self.sess.run([self.deepayni_optimizer, self.decrypt_err_reciver],feed_dict = {self.msg: msg_in_val, self.key: key_val})

        deepayni_decrypt_error = min(deepayni_decrypt_error, decrypt_err)

        if network == 'listener':

            _, decrypt_err = self.sess.run([self.prakash_optimizer, self.decrypt_err_listener],
                                       feed_dict={self.msg: msg_in_val, self.key: key_val})
        prakash_decrypt_error = min(prakash_decrypt_error, decrypt_err)

        return deepayni_decrypt_error, prakash_decrypt_error

    def plot_errors(self):
        sns.set_style("darkgrid")
        plt.plot(self.deepayni_errors)
        plt.plot(self.prakash_errors)
        plt.legend(['Reciever', 'listener'])
        plt.xlabel('Epoch')
        plt.ylabel('Lowest Decryption error achiprakashd')
        plt.show()
