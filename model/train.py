import sys
import tensorflow as tf
import numpy as np
from model import RINet
import os
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.metrics import roc_auc_score
from keras import backend as K
np.random.seed(2018)

#configuration
cf_max_orderlen = 100
cf_max_productlen = 145
cf_order_feanum = 3
cf_depthlist = [8, 25, 31]
cf_attention_size = 100
cf_state_size = 100
cf_batch_size = 5
cf_learning_rate = 0.001
cf_num_epochs = 5
cf_validate_every = 1
cf_ckpt_dir = "../model_path/"

def generate_batch(trainX, trainY, batch_size):
    batch_x = []
    batch_y = []
    curr_num = 0
    while 1:
        totalnum = len(trainX)
        index = np.arange(totalnum)
        np.random.shuffle(index)
        tempX = []
        tempY = []
        for key in index:
            tempX.append(trainX[key])
            tempY.append(trainY[key])
        trainX = tempX
        trainY = tempY
        for i in range(totalnum):
            curr_num += 1
            batch_x.append(trainX[i])
            batch_y.append(trainY[i])
            if curr_num == batch_size:
                yield batch_x, np.asarray(batch_y)
                batch_x = []
                batch_y = []
                curr_num = 0

def shuffle_split(train_x, train_y, ratio=0.1):
    index = np.arange(len(train_x))
    np.random.shuffle(index)
    tempX = []
    tempY = []
    for key in index:
        tempX.append(train_x[key])
        tempY.append(train_y[key])
    train_x = tempX
    train_y = tempY

    train_num = int(len(train_x)*(1-ratio))
    t_x = train_x[0:train_num]
    t_y = train_y[0:train_num]
    v_x = train_x[train_num:-1]
    v_y = train_y[train_num:-1]
    return t_x, t_y, v_x, v_y

def padding_senquence(input, maxlen):
    total_num = len(input)
    padding_size = (total_num, maxlen[0], maxlen[1])
    result = np.zeros(padding_size)
    for i, usr_sessions in enumerate(input):
        for j, session in enumerate(usr_sessions):
            if j < maxlen[0]:
                for k, activity in enumerate(session):
                    if k < maxlen[1]:
                        result[i,j,k] = activity
    return result

def get_batch_train_data(batch_x, user_dict):
    #batch order_fea
    order_len = [] 
    order_productfea = []
    order_productid = []
    order_departmentid = []
    order_aisleid = []
    order_productidx = [] 
    
    #batch product_fea
    curr_productfea = []
    curr_productid = []
    curr_aisleid = []
    curr_departmentid = [] 
    curr_productidx = []
    for i in range(len(batch_x)):
        curr_usr = batch_x[i][0]
        order_len.append(user_dict[curr_usr][0])
        order_productfea.append(user_dict[curr_usr][1])
        order_productid.append(user_dict[curr_usr][2])
        order_departmentid.append(user_dict[curr_usr][3])
        order_aisleid.append(user_dict[curr_usr][4])
        order_productidx.append(user_dict[curr_usr][5])

        curr_productfea.append(batch_x[i][1])
        curr_productid.append(batch_x[i][2])
        curr_aisleid.append(batch_x[i][3])
        curr_departmentid.append(batch_x[i][4])
        curr_productidx.append(batch_x[i][5])
    #padding
    order_productfea = padding_senquence(order_productfea,maxlen=[100, 3])
    order_productid = padding_senquence(order_productid, maxlen=[100, 145])
    order_aisleid = padding_senquence(order_aisleid, maxlen=[100, 145])
    order_departmentid = padding_senquence(order_departmentid, maxlen=[100, 145])
    order_productidx = padding_senquence(order_productidx, maxlen=[100, 145])

    order_len = np.asarray(order_len)
    curr_productfea = np.asarray(curr_productfea)
    curr_productid = np.asarray(curr_productid)
    curr_aisleid = np.asarray(curr_aisleid)
    curr_departmentid = np.asarray(curr_departmentid)
    curr_productidx = np.asarray(curr_productidx)
    #checking 
    print(order_productfea.shape)
    print(order_productid.shape)
    print(order_aisleid.shape) 
    print(order_departmentid.shape) 
    print(order_productidx.shape) 
    print(order_len.shape) 

    print(curr_productfea.shape)
    print(curr_productid.shape) 
    print(curr_aisleid.shape)
    print(curr_departmentid.shape)  
    print(curr_productidx.shape) 
    return order_len, order_productfea, order_productid, order_departmentid, order_aisleid, \
           order_productidx, curr_productfea, curr_productid, curr_aisleid, curr_departmentid, curr_productidx


def main(_):
    #load data(user_dict, train_data)
    with open("../data/processed/final_user_data","rb") as f_in:
        user_dict = pickle.load(f_in)
    with open("../data/processed/final_train_data","rb") as f_in:
        train_data_list, train_data_label = pickle.load(f_in)
        train_data, train_label, test_data, test_label = shuffle_split(train_data_list, train_data_label)
    print("finish load data")
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        recurrIntereNet = RINet(cf_max_orderlen, cf_max_productlen, cf_order_feanum, cf_depthlist,
                                 cf_attention_size, cf_state_size, cf_learning_rate)

        print("finish model")

        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists("../model_path/"+"checkpoint"):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess,"../model_path/model.ckpt")
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
        
        if "train_history" in os.listdir("./") :
            with open("train_history", 'rb') as file_in:
                step_list, loss_list, test_step_list, test_loss_list = pickle.load(file_in)
        else:
            step_list = []
            loss_list = []
            test_step_list = []
            test_loss_list = []

        curr_epoch=sess.run(recurrIntereNet.epoch_step)
        #3.feed data & training
        #number_of_training_data=len(trainX)
        batch_size = cf_batch_size
        number_of_training_data=len(train_data)
        train_gendata = generate_batch(train_data, train_label, batch_size)
        early_stop = False
        for epoch in range(curr_epoch, cf_num_epochs):
            loss, counter = 0.0, 0
            #trainX, trainY = label_shuffle(orignal_X, orignal_Y)
            print("number of training data: %d"%number_of_training_data)
            for i in range(int(number_of_training_data / batch_size) + 1):
            #for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)): 
                batch_x, batch_y = next(train_gendata)
                order_len, order_productfea, order_productid, order_departmentid, order_aisleid, \
                order_productidx, curr_productfea, curr_productid, curr_aisleid, \
                curr_departmentid, curr_productidx = get_batch_train_data(batch_x, user_dict)

                curr_loss, _, step_num = sess.run([recurrIntereNet.losses, recurrIntereNet.train_op, recurrIntereNet.global_step], feed_dict={
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
                                                recurrIntereNet.label:batch_y})

                loss, counter = loss+curr_loss, counter+1
                step_list.append(step_num)
                loss_list.append(curr_loss)
                if counter %10==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f" %(epoch,counter,loss/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
            #epoch increment
                if step_num % 50 == 0:
                    test_step_list.append(step_num)
                    eval_loss = do_eval(sess, recurrIntereNet, test_data, test_label, batch_size)
                    print("Epoch %d Validation Loss:%.4f" % (epoch,eval_loss))
                    test_loss_list.append(eval_loss)
                    '''
                    if eval_trueloss <= 0.035:
                        save_path=FLAGS.ckpt_dir+"model.ckpt"
                        saver.save(sess,save_path,global_step=epoch)
                        early_stop = True
                        break
                    '''
            '''
            if early_stop == True:
                break
            '''
            with open("train_history",'wb') as file_out:
                pickle.dump([step_list, loss_list, test_step_list, test_loss_list], file_out)
                   

            print("going to increment epoch counter....")
            sess.run(recurrIntereNet.epoch_increment)
            # 4.validation
            print(epoch,cf_validate_every,(epoch % cf_validate_every==0))
            if epoch % cf_validate_every==0:
                #eval_loss, eval_trueloss, eval_acc=do_eval(sess,textRNN,testX,testY,batch_size)
                #print("Epoch %d Validation Loss:%.3f\tValidation trueLoss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_trueloss,eval_acc))
                #save model to checkpoint
                save_path= cf_ckpt_dir + "model.ckpt"
                saver.save(sess,save_path,global_step=epoch)
        '''
        plt.plot(step_list, loss_list)
        plt.savefig("noL2_drop_maxpool_train")
        plt.close()
        plt.plot(test_step_list, test_loss_list)
        plt.savefig("noL2_drop_maxpool_test")
        plt.close()
        '''

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        #test_loss, test_acc = do_eval(sess, textRNN, testX, testY, batch_size,vocabulary_index2word_label)
    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, recurrIntereNet, test_data, test_label, user_dict, batch_size):
    number_examples=len(test_data)
    start = 0
    end = number_examples
    eval_loss = 0.0
    while(start < end):
        if start+batch_size < end:
            batch_x, batch_y = test_data[start:start+batch_size], test_label[start:start+batch_size]
            batch_y = np.asarray(batch_y)

            order_len, order_productfea, order_productid, order_departmentid, order_aisleid, \
            order_productidx, curr_productfea, curr_productid, curr_aisleid, \
            curr_departmentid, curr_productidx = get_batch_train_data(batch_x, user_dict)

            curr_loss = sess.run(recurrIntereNet.losses, feed_dict={
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
                                            recurrIntereNet.label:batch_y})          

            eval_loss = eval_loss + curr_loss*batch_size
        else:
            batch_x, batch_y = test_data[start:end], test_label[start:end]
            batch_y = np.asarray(batch_y)

            order_len, order_productfea, order_productid, order_departmentid, order_aisleid, \
            order_productidx, curr_productfea, curr_productid, curr_aisleid, \
            curr_departmentid, curr_productidx = get_batch_train_data(batch_x, user_dict)

            curr_loss = sess.run(recurrIntereNet.losses, feed_dict={
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
                                            recurrIntereNet.label:batch_y})                
            rest_count = end - start
            eval_los = eval_loss + curr_loss*rest_count 
        #logits = tf.sigmoid(logits)
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        
        start = start+batch_size 
    return eval_loss/float(end)

if __name__ == "__main__":
    tf.app.run()