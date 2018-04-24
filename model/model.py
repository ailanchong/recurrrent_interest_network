import tensorflow as tf
import numpy as np
from tf_utils import generate_allwindow, lstm_block
class RINet(object):
    def __init__(self, max_orderlen, max_productlen, order_feanum, depthlist, attention_size, state_size):
        self.max_orderlen = max_orderlen
        self.order_feanum = order_feanum
        self.max_productlen = max_productlen
        self.depthlist = depthlist
        self.attention_size = attention_size
        self.state_size = state_size
        self.instantiate_weights()
        #order_info
        self.order_productfea = tf.placeholder(tf.int32, [None, self.max_orderlen, order_feanum], name="order_productfea")
        self.order_productid = tf.placeholder(tf.int32, [None, self.max_orderlen, self.max_productlen], name="order_productid")
        self.order_aisleid = tf.placeholder(tf.int32, [None, self.max_orderlen, self.max_productlen], name="order_aisleid")
        self.order_departmentid = tf.placeholder(tf.int32,[None, self.max_orderlen, self.max_productlen], name="order_departmentid")
        self.order_productidx = tf.placeholder(tf.int32,[None, self.max_orderlen, self.max_productlen], name="order_productidx")
        self.order_len = tf.placeholder(tf.int32, [None], name="order_len")
        #recommend product
        self.curr_productfea = tf.placeholder(tf.int32, [None, order_feanum], name="curr_productfea")
        self.curr_productid = tf.placeholder(tf.int32, [None],name="curr_productid")
        self.curr_aisleid = tf.placeholder(tf.int32, [None], name="curr_aisleid")
        self.curr_departmentid = tf.placeholder(tf.int32, [None], name="curr_departmentid")
        self.curr_productidx = tf.placeholder(tf.int32, [None], name="curr_productidx")
        #label
        self.label = tf.placeholder(tf.int32,[None], name="label")
        self.inference()
    def instantiate_weights(self):
        """define all weights here"""
        self.product_embeddings = tf.get_variable(
            name='product_embeddings',
            shape=[50000, 300],
            dtype=tf.float32
        )
        self.aisle_embeddings = tf.get_variable(
            name='aisle_embeddings',
            shape=[250, 50],
            dtype=tf.float32
        )
        self.department_embeddings = tf.get_variable(
            name='department_embeddings',
            shape=[50, 10],
            dtype=tf.float32
        )
        self.W_relu = tf.get_variable("W_relu",shape=[545, 30]) 
        self.b_relu = tf.get_variable("bias_relu",shape=[30])       
        self.W_projection = tf.get_variable("W_projection",shape=[30, 1]) 
        self.b_projection = tf.get_variable("bias_projection",shape=[1])    
    def one_hot_translate(self, input, depth):
        """
        input = [batch_size, max_orderlen, max_productlen]
        output = [batch_size, max_orderlen, max_productlen, one_hot_len]
        """
        output = tf.one_hot(input, depth)
        return output

    def one_hot_translate_list(self, input, depthlist):
        """
        input = [batch_size, max_orderlen, order_feanum]
        depthlist = [order_feanum] to record the one_hot depth for every feature.
        output = [batch_size, max_orderlen, max_productlen, sum(order_feanum)] 
        """
        input = tf.expand_dims(input, 2)
        input = tf.tile(input,[1, 1, self.max_productlen, 1])
        result = []
        for i in range(len(depthlist)):
            curr_input = input[:,:,:,i]
            #curr_input = tf.squeeze(curr_input,-1) #[batch_size, max_orderlen, max_productlen]
            curr_output = self.one_hot_translate(curr_input, depthlist[i])
            result.append(curr_output)
        #result =[curr_output], curr_output=[batch_size, max_orderlen, max_productlen, depth]
        output = tf.concat(result, axis=-1)
        return output 
    def one_hot_translate_product(self, input, depthlist):
        result = []
        for i in range(len(depthlist)):
            curr_input = input[:,i]
            curr_output = self.one_hot_translate(curr_input, depthlist[i])
            result.append(curr_output)
        output = tf.concat(result, axis=-1)
        return output

    def get_user_input(self):
        """
        to get the user_input = [batch_size, max_orderlen, max_productlen, product_vec]
        """
        result = []
        result.append(self.one_hot_translate_list(self.order_productfea, self.depthlist))
        result.append(tf.nn.embedding_lookup(self.product_embeddings, self.order_productid))
        result.append(tf.nn.embedding_lookup(self.aisle_embeddings, self.order_aisleid))
        result.append(tf.nn.embedding_lookup(self.department_embeddings, self.order_departmentid))
        result.append(self.one_hot_translate(self.order_productidx, self.max_productlen+1))
        user_input = tf.concat(result,axis=-1)
        return user_input

    def get_product_input(self):
        """
        to get the product_input =[batch_size, product_vec]
        """
        result = []
        result.append(self.one_hot_translate_product(self.curr_productfea, self.depthlist))
        result.append(tf.nn.embedding_lookup(self.product_embeddings, self.curr_productid))
        result.append(tf.nn.embedding_lookup(self.aisle_embeddings, self.curr_aisleid))
        result.append(tf.nn.embedding_lookup(self.department_embeddings, self.curr_departmentid))
        result.append(self.one_hot_translate(self.curr_productidx, self.max_productlen+1))
        product_input = tf.concat(result, axis=-1)
        return product_input
    def inference(self):
        user_input = self.get_user_input()
        product_input = self.get_product_input()
        allwindow = generate_allwindow(user_input, product_input, self.attention_size)
        lstm_output = lstm_block(allwindow, product_input, self.order_len, self.state_size)
        temp = tf.nn.relu(tf.matmul(lstm_output, self.W_relu) + self.b_relu)
        logits = tf.matmul(temp, self.W_projection) + self.b_projection
        self.logits = logits
    def loss(self):
        with tf.name_scope("loss"):
            losses=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,logits=self.logits)
            return losses



def test():
    max_orderlen = 100
    max_productlen = 20
    order_feanum = 3
    depthlist = [31, 8, 25]
    attention_size = 100
    state_size = 100
    batch_size = 10
    recurrIntereNet = RINet(max_orderlen, max_productlen, order_feanum, depthlist, attention_size, state_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            order_productfea = np.zeros([10, max_orderlen, order_feanum])
            order_productid = np.zeros([10, max_orderlen, max_productlen])
            order_aisleid = np.zeros([10, max_orderlen, max_productlen])
            order_departmentid = np.zeros([10, max_orderlen, max_productlen])
            order_productidx = np.zeros([10, max_orderlen, max_productlen])
            order_len = np.ones([10])
            #recommend product
            curr_productfea = np.zeros([10, order_feanum])
            curr_productid = np.zeros([10])
            curr_aisleid = np.zeros([10])
            curr_departmentid = np.zeros([10])
            curr_productidx = np.zeros([10])            
            #label
            label = np.zeros([10])
            
            logits = sess.run(recurrIntereNet.logits,feed_dict={
                                            recurrIntereNet.order_productfea:order_productfea,
                                            recurrIntereNet.order_productid : order_productid,
                                            recurrIntereNet.order_aisleid : order_aisleid,
                                            recurrIntereNet.order_departmentid: order_departmentid,
                                            recurrIntereNet.order_productidx: order_productidx,
                                            recurrIntereNet.order_len:order_len,
                                            recurrIntereNet.curr_productfea:curr_productfea,
                                            recurrIntereNet.curr_productid: curr_productid,
                                            recurrIntereNet.curr_aisleid:curr_aisleid,
                                            recurrIntereNet.curr_departmentid: curr_departmentid,
                                            recurrIntereNet.curr_productidx: curr_productidx,
                                            recurrIntereNet.label:label})
            print(np.asarray(logits).shape)       

if __name__ == "__main__":
    test()







