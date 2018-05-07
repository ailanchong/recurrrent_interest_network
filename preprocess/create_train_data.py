import os
import pandas as pd

#print(df.shape)
#print(df['reordered'].sum())
#print(train_products['reordered'].sum())
if __name__ == "__main__":
    orders = pd.read_csv('../data/raw/orders.csv')
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    train_products = pd.read_csv('../data/raw/order_products__train.csv')
    products = pd.read_csv('../data/raw/products.csv')
    test_orders = orders[orders.eval_set == 'test']
    train_orders = orders[orders.eval_set == 'train']
    prior_orders = orders[orders.eval_set == 'prior']

    #get all products that every user has purchased [user_id, product_id]
    df = prior_orders.merge(prior_products, how="inner", on="order_id")
    df = df[['user_id', 'product_id']]
    df = df[~df.duplicated()]
    #get all products that train user has purchased [user_id, product_id]
    df = df.merge(train_orders, how="inner", on="user_id")
    df = df.merge(train_products, how="left", on=["order_id", "product_id"])
    df.drop(['order_id','eval_set','order_number'],inplace=True,axis=1)
    df.add_to_cart_order = 0
    df['reordered'] = df['reordered'].fillna(0).astype(int)
    df = df.merge(products, how="left", on="product_id")
    df.drop("product_name", inplace=True, axis=1)
    df.order_dow = df.order_dow.map(str)
    df.order_hour_of_day = df.order_hour_of_day.map(str)
    df.days_since_prior_order = df.days_since_prior_order.map(str)
    df['curr_productfea'] = (df.order_dow + " " + df.order_hour_of_day + " " + df.days_since_prior_order)
    df.drop(['order_dow','order_hour_of_day','days_since_prior_order'], inplace=True, axis=1)
    df.to_csv('../data/processed/train_data.csv', index=False)
    print(df.head())
    print(df.shape)