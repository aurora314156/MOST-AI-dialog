import numpy as np
import sys
import logging
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Activation, Embedding, Bidirectional, GRU



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

def test(word2vecPath):
    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # into our TensorFlow and Keras models
    model = Word2Vec.load(word2vecPath)
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    # setup the embedding layer
    embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix])

def w2vToEmbeddingLayer(embeddingDim, embeddingMatrix, word2vecPath):

    # model.add(Embedding(len(embeddingMatrix),
    #                         input_length = embeddingDim,
    #                         output_dim = embeddingDim,
    #                         weights=[embeddingMatrix],
    #                         trainable=False))
    print("Start w2vToEmbeddingLayer")
    word2vecmodel = Word2Vec.load(word2vecPath)
    model = Sequential()
    model.add(word2vecmodel.wv.get_keras_embedding(train_embeddings=False))
    model.add(Bidirectional(GRU(units = 50)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.summary()
    #model.fit(embeddingMatrix,embeddingMatrix,epochs=5,batch_size=32)
    print("train..")

def main():
    
    embeddingDim = 300
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    word2vecPath = sys.argv[1]
    embeddingMatrix = w2vToEmbeddingMatrix(word2vecPath)
    w2vToEmbeddingLayer(embeddingDim, embeddingMatrix, word2vecPath)
    
if __name__ == "__main__":
    main()