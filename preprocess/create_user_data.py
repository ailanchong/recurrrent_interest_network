import os

import pandas as pd


def parse_order(x):
    series = pd.Series()

    series['products'] = '_'.join(x['product_id'].values.astype(str).tolist())
    series['aisles'] = '_'.join(x['aisle_id'].values.astype(str).tolist())
    series['departments'] = '_'.join(x['department_id'].values.astype(str).tolist())
    series['order_indexes'] = '_'.join(x['add_to_cart_order'].values.astype(str).tolist())
    series['order_fea'] = '_'.join(map(str,[x['order_dow'].iloc[0], x['order_hour_of_day'].iloc[0], 
                                    x['days_since_prior_order'].iloc[0]]))
    return series


def parse_user(x):
    parsed_orders = x.groupby('order_id', sort=False).apply(parse_order)
    series = pd.Series()
    series['order_fea'] = ' '.join(parsed_orders['order_fea'].values.astype(str).tolist())
    series['product_ids'] = ' '.join(parsed_orders['products'].values.astype(str).tolist())
    series['aisle_ids'] = ' '.join(parsed_orders['aisles'].values.astype(str).tolist())
    series['department_ids'] = ' '.join(parsed_orders['departments'].values.astype(str).tolist())
    series['product_add_orders'] = ' '.join(parsed_orders['order_indexes'].values.astype(str).tolist())

    return series

if __name__ == '__main__':
    orders = pd.read_csv('../data/raw/orders.csv')
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    products = pd.read_csv('../data/raw/products.csv')
    df = orders.merge(prior_products, how='inner', on='order_id')
    df = df.merge(products, how='inner', on='product_id')
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0).astype(int)

    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')
    #df = df[df.user_id == 1]
    #print(df.head)
    user_data = df.groupby('user_id', sort=False).apply(parse_user).reset_index()
    user_data.to_csv('../data/processed/user_data.csv', index=False)
    