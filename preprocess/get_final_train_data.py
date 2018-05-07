import pandas as pd 
import numpy as np  
import pickle 

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

def self_split(input):
    """
    input is str, format: 2_8_0 3_7_15 3_12_21 4_7_29 4_15_28
    split the input to [[]]
    """
    input = input.strip().split(" ")
    result = []
    for key in input:
        key = key.split("_")
        key = list(map(int, key))
        if(len(key) == 3):
            if(key[0] == 0):
                key[0] = 7
            if(key[1] == 0):
                key[1] = 24
        result.append(key)
    return result


#padding 
'''
order_productfea = padding_senquence(order_productfea,maxlen=[100, 3])
order_productid = padding_senquence(order_productid, maxlen=[100, 145])
order_aisleid = padding_senquence(order_aisleid, maxlen=[100, 145])
order_departmentid = padding_senquence(order_departmentid, maxlen=[100, 145])
order_productidx = padding_senquence(order_productidx, maxlen=[100, 145])
'''
#check
"""
order_len = np.asarray(order_len)

curr_productfea = np.asarray(curr_productfea)
curr_productid = np.asarray(curr_productid)
curr_aisleid = np.asarray(curr_aisleid)
curr_departmentid = np.asarray(curr_departmentid)
curr_productidx = np.asarray(curr_productidx)
label = np.asarray(label)

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
print(label.shape)
"""

if __name__ == "__main__":
    train_data = pd.read_csv('../data/processed/train_data.csv')
    print(train_data.shape)
    print(train_data.head())
    #recommend product
    curr_productfea = [] 
    curr_productid = []
    curr_aisleid = []
    curr_departmentid = [] 
    curr_productidx = []

    train_data_list = []
    count = 0
    for row in train_data.itertuples():
        count += 1
        if(count % 10000 == 0):
            print("curr count:{}".format(count))
            #break
        # add curr product_info
        temp_train_data = []
        temp_train_data.append(row.user_id) #第0维 user_id

        temp_productfea = row.curr_productfea
        temp_productfea = temp_productfea.split(" ")
        temp_productfea = list(map(int,list(map(float, temp_productfea))))
        if(temp_productfea[0] == 0):
            temp_productfea[0] = 7
        if(temp_productfea[1] == 0):
            temp_productfea[1] = 24
        #print(temp_productfea)
        temp_train_data.append(temp_productfea) #第1维 productfea 
        temp_train_data.append(row.product_id)   #第2维 product_id
        temp_train_data.append(row.aisle_id) # 第3维 aisle_id
        temp_train_data.append(row.department_id) #第4维 department_id
        temp_train_data.append(0) #第5维 product_idx

        train_data_list.append(temp_train_data)

    ##save
    with open("../data/processed/final_train_data","wb") as f_out:
        pickle.dump(train_data_list, f_out, protocol=4)
