import numpy as np
import sys
import logging
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding


def w2vToEmbeddingMatrix(word2vecPath):
    # load model
    model = Word2Vec.load(word2vecPath)
    
    word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    print(len(vocab_list))
    j = 0
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        embeddings_matrix[i + 1] = vocab_list[i][1]
        j+=1
        #if j % 100000:
        #    print(j)
    return embeddings_matrix

def w2vToEmbeddingLayer(embeddingDim, embeddingMatrix):
    
    print("Start w2vToEmbeddingLayer")
    embedding_layer = Embedding(len(embeddingMatrix),
                            input_dim = embeddingDim,
                            weights=[embeddingMatrix])
    print("Success")

def main():
    
    embeddingDim = 300
    embeddingMatrix = w2vToEmbeddingMatrix(word2vecPath = sys.argv[1])
    w2vToEmbeddingLayer(embeddingDim, embeddingMatrix)
    
if __name__ == "__main__":
    main()