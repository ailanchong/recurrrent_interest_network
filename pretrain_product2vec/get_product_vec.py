import pandas as pd
import numpy as np
import gensim
import pickle
def get_word2vec(sentences, embedding_dim, window_size, total_num):
    model = gensim.models.Word2Vec(sentences, size=embedding_dim, window=window_size, min_count=1, workers=4)
    embeddings_index = {}
    print(len(model.wv.vocab))
    for i in model.wv.vocab.keys():
        embeddings_index[i] = model[i]

    print('Total %s activity vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((total_num, embedding_dim))
    for id, embedding in embeddings_index.items():
        id = int(id)
        embedding_matrix[id] = embedding
    return embedding_matrix
def get_sentence(train_orders, prior_orders, key):
    train_orders[key] = train_orders[key].astype(str)
    prior_orders[key] = prior_orders[key].astype(str)    
    train_items = train_orders.groupby("order_id").apply(lambda order: order[key].tolist())
    prior_items = prior_orders.groupby("order_id").apply(lambda order: order[key].tolist())
    sentences = prior_items.append(train_items)
    print(sentences.shape)
    print(sentences.head())
    longest = np.max(sentences.apply(len))
    sentences = sentences.values
    return sentences, longest          

def get_embedding_matrix(train_orders, prior_orders, key, embedding_dim, total_num):
    sentences, longest = get_sentence(train_orders, prior_orders, key)
    embedding_matrix = get_word2vec(sentences, embedding_dim, longest, total_num)
    return embedding_matrix



if __name__ == "__main__":
    train_orders = pd.read_csv("../data/raw/order_products__train.csv")
    prior_orders = pd.read_csv("../data/raw/order_products__prior.csv")
    products = pd.read_csv("../data/raw/products.csv")
    train_orders = train_orders.merge(products, how='inner', on='product_id')
    prior_orders = prior_orders.merge(products, how='inner', on='product_id')

    product_matrix = get_embedding_matrix(train_orders, prior_orders, "product_id", 50, 50000) #49686
    department_matrix = get_embedding_matrix(train_orders, prior_orders, "department_id", 5, 50)
    aisle_matrix = get_embedding_matrix(train_orders, prior_orders, "aisle_id", 10, 250)
    with open("../data/processed/word_matrix", "wb") as f_out:
        pickle.dump([product_matrix, department_matrix, aisle_matrix], f_out, protocol=4)

