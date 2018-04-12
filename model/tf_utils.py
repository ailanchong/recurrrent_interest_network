import tensorflow as tf
import numpy as np


def interest_block(order_input, product_input, attention_size):
    hidden_size = 3 * order_input.shape[1]
    #trainable parameters
    W_1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_1 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    W_2 = tf.Variable(tf.random_normal([attention_size, 1]), stddev=0.1)
    b_2 = tf.Variable(tf.random_normal([1],stddev=0.1))
    #concate infomation
    product_input = tf.expand_dims(product_input, 0)
    product_input = tf.tile(product_input, [order_input.shape[0],1])
    point_wise_temp = tf.multiply(product_input, order_input)
    total_info = tf.concate([order_input, point_wise_temp, product_input], axis=1)
    #generate weight
    middle_layer = tf.relu(tf.matmul(total_info, W_1) + b_1)
    result_weight = tf.matmul(middle_layer, W_2) + b_2
    return result_weight

def timewindow_block(order_input, product_input, attention_size):
    weight = interest_block(order_input, product_input, attention_size)
    



    


