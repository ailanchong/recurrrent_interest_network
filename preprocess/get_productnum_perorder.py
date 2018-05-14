import os

import pandas as pd


def parse_order(x):
    series = pd.Series()
    series['product_num'] = x['add_to_cart_order'].max()                                
    return series


def parse_user(x):
    parsed_orders = x.groupby('order_id', sort=False).apply(parse_order)
    series = pd.Series()

    series['product_num'] = parsed_orders['product_num'].mean()

    return series

if __name__ == '__main__':
    orders = pd.read_csv('../data/raw/orders.csv')
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    df = orders.merge(prior_products, how='inner', on='order_id')
    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')
    #df = df[df.user_id == 1]
    #print(df.head)
    user_data = df.groupby('user_id', sort=False).apply(parse_user).reset_index()
    print(user_data['product_num'].max())
    print(user_data['product_num'].mean())
    user_data.to_csv('../data/processed/user_productnum.csv', index=False)
    