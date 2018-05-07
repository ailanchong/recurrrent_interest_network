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



if __name__ == "__main__":
    user_data = pd.read_csv('../data/processed/user_data.csv')
    #order_info
    order_productfea = []
    order_productid = []
    order_aisleid = []
    order_departmentid = []
    order_productidx = []
    order_len = []
    print(user_data.shape)
    user_dict = {}
    
    count = 0
    for row in user_data.itertuples():
        # add order_info
        temp_user_list = []
        curr_order_productfea = row.order_fea
        curr_order_productfea = self_split(curr_order_productfea)
        temp_user_list.append(len(curr_order_productfea))  #第0维 order_len
        temp_user_list.append(curr_order_productfea) #第1维 order_productfea

        curr_order_productid = row.product_ids
        curr_order_productid = self_split(curr_order_productid)
        temp_user_list.append(curr_order_productid) #第2维 order_productid

        curr_order_departmentid = row.department_ids
        curr_order_departmentid = self_split(curr_order_departmentid)
        temp_user_list.append(curr_order_departmentid) #第3维 order_departmentid

        curr_order_aisleid = row.aisle_ids
        curr_order_aisleid = self_split(curr_order_aisleid)
        temp_user_list.append(curr_order_aisleid) #第4维 order_aisleid

        curr_order_productidx = row.product_add_orders
        curr_order_productidx = self_split(curr_order_productidx)
        temp_user_list.append(curr_order_productidx) #第5维 order_productidx
        
        curr_user = row.user_id
        user_dict[curr_user] = temp_user_list
        count += 1 
        if(count % 10000 == 0):
            print("curr count:{}".format(count))
            #print(temp_user_list)
               
    ##save
    with open("../data/processed/final_user_data","wb") as f_out:
        pickle.dump(user_dict, f_out, protocol=4)