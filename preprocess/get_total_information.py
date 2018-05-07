import pandas as pd

if __name__ == "__main__":
    orders = pd.read_csv('../data/raw/orders.csv')
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    print(orders.order_number.max())  #max order number is 100
    print(prior_products.add_to_cart_order.max()) #max product number is 145
      