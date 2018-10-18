import numpy as np
import sys
import logging
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Activation, Embedding, Bidirectional, GRU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


def getEmbeddingWeights(word2vecPath, tokenizer, embedding_dim):

    # load word2vec model
    model = gensim.models.Word2Vec.load(word2vecPath)

    embedding_matrix = np.zeros((max_num_words, embedding_dim))
    
    for word, i in tokenizer.word_index.items():
        if word in model.wv.vocab and i < max_num_words:
            embedding_vector = model.wv.syn0[model.wv.vocab[word].index]
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def w2vToEmbeddingLayer(maxNumWords, embeddingDim, embeddingWeights):

    print("Start w2vToEmbeddingLayer")
    word2vecmodel = Word2Vec.load(word2vecPath)
    model = Sequential()
    embeddingLayer = Embedding(input_dim=MAX_NUM_WORDS,
                            output_dim=EMBEDDING_DIM,
                            input_length=MAX_SEQ_LENGTH,
                            weights=[embeddingWeights],
                            trainable=False
                           )
    #model.add(Bidirectional(GRU(units = 50)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.summary()
    #model.fit(embeddingMatrix,embeddingMatrix,epochs=5,batch_size=32)
    print("train..")

def main():
    
    embeddingDim = 300
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    embeddingWeights = getEmbeddingWeights(word2vecPath = sys.argv[1], )
    w2vToEmbeddingLayer(maxNumWords, embeddingDim, embeddingWeights)
    
if __name__ == "__main__":
    main()
    