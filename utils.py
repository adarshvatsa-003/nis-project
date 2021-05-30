import numpy as np
import tensorflow as tf
from config import MSG_LEN,KEY_LEN,BATCH_SIZE

# Function to generate n random messages and keys
def gen_data(n=BATCH_SIZE, msg_len=MSG_LEN, key_len=KEY_LEN):
    return (np.random.randint(0, 2, size=(n, msg_len)) * 2 - 1),(np.random.randint(0, 2, size=(n, key_len)) * 2 - 1)


def init_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

