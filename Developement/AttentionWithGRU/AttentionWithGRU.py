import numpy as np
import sys
sys.path.append('../')
import logging
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Activation, Embedding, Bidirectional, GRU
from keras.preprocessing.text import Tokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


class AttentionWithGRU():
    def __init__(self, CQADataSet, w2vmodel, embeddingDim):
        self.CQADataSet = CQADataSet
        self.w2vmodel = w2vmodel
        self.embeddingDim = embeddingDim

    def AttentionWithGRUMain(self):
        
        questionWordList = self.CQADataSet[0].getQuesiton()
        # forward GRU embedding weights
        embeddingWeights = getEmbeddingWeights(questionList)
        
    def getEmbeddingWeights(self, wordOfSentence):
        
        # for text tokenizer
        tokenizer = Tokenizer(num_words=len(sentence))
        textList = []
        for s in sentence:
            textList.append(s)
        tokenizer.fit_on_texts(textList)
        # create np matrix for 
        embedding_matrix = np.zeros((len(sentence), self.embeddingDim))
        # read word vector from model, then save to np matrix 
        for word, i in tokenizer.word_index.items():
            if word in self.w2vmodel.wv.vocab and i < max_num_words:
                embedding_vector = self.w2vmodel.wv.syn0[self.w2vmodel.wv.vocab[word].index]
                embedding_matrix[i] = embedding_vector

        return embeddingMatrix

    def w2vTOEmbeddingLayer(self, maxNumWords, embeddingDim, embeddingWeights):

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