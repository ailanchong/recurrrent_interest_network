import tensorflow as tf
import numpy as np

def interest_block(order_input, product_input, attention_size=10):
    """
    input:
    order_input = [batch_size, product_num, product_vec]
    product_input = [batch_size, product_vec]
    output:
    result_weight = [batch_size, product_num, 1]
    """
    hidden_size = 3 * order_input.shape[2].value
    print(hidden_size)
    print(attention_size)
    #trainable parameters
    W_1 = tf.get_variable("W_1", shape = [hidden_size, attention_size])
    b_1 = tf.get_variable("b_1", shape = [attention_size])
    W_2 = tf.get_variable("W_2", shape = [attention_size, 1])
    b_2 = tf.get_variable("b_2", shape = [1])
    #concate infomation
    product_input = tf.expand_dims(product_input, 1)
    product_input = tf.tile(product_input, [1,order_input.shape[1],1])
    point_wise_temp = tf.multiply(product_input, order_input)
    total_info = tf.concat([order_input, point_wise_temp, product_input], axis=2)
    #generate weight
    middle_layer = tf.nn.relu(tf.tensordot(total_info, W_1, axes=[2, 0]) + b_1)
    result_weight = tf.tensordot(middle_layer, W_2, axes=[2, 0]) + b_2
    return result_weight

def timewindow_block(order_input, product_input, attention_size):
    """
    input:
    order_input = [batch_size, product_num, product_vec]
    product_input = [batch_size, product_vec]
    output:
    order_deal = [batch_size, product_vec]
    """
    hidden_size = order_input.shape[2]
    weight = interest_block(order_input, product_input, attention_size) #[batch_size, product_num, 1]
    weight = tf.tile(weight, [1,1,hidden_size])
    order_deal = tf.multiply(order_input, weight)
    order_deal = tf.reduce_mean(order_deal, 1)
    return order_deal

def generate_allwindow(user_input, product_input, attention_size):
    """
    input:
    user_input = [batch_size, order_num, product_num, product_vec]
    product_input = [batch_size, product_vec]
    output:
    allwindow = [batch_size, order_num, product_vec]
    """
    with tf.variable_scope("time_window"):
        hidden_size = 3 * user_input.shape[3].value
        W_1 = tf.get_variable("W_1", shape = [hidden_size, attention_size])
        b_1 = tf.get_variable("b_1", shape = [attention_size])
        W_2 = tf.get_variable("W_2", shape = [attention_size, 1])
        b_2 = tf.get_variable("b_2", shape = [1])
    with tf.variable_scope("time_window",reuse=True):
        allwindow = []
        for i in range(user_input.shape[1]):
            order_input = user_input[:,i,:,:]
            allwindow.append(timewindow_block(order_input, product_input, attention_size))
        return tf.stack(allwindow, axis=1)
def lstm_block(allwindow, product_input, lengths, state_size, keep_prob=1.0, scope='lstm_block', reuse=False):
    """
    input:
    allwindow = [batch_size, order_num, product_vec]
    product_input = [batch_size, product_vec]
    lenghts = [batch_size]
    output:
    output_fea = [batch_size, hidden_size + product_vec]
    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=allwindow,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )#outputs:[batch_size, hidden_size]
        outputs = outputs[:,-1,:]
        result = tf.concat([outputs, product_input],axis=-1)
        return result

 


        









    


