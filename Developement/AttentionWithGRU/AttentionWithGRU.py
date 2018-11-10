import os
import sys
import tensorflow as tf
import logging
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AttentionWithGRU():
    def __init__(self, CQADataSet, tqn):
        self.CQADataSet = CQADataSet
        self.tqn = tqn

    def AttentionWithGRUMain(self):
        
        guessAnsList = []
        questionWordList = self.CQADataSet[0].getQuestion()
        
        # QuestionBidirectionalGRU
        self.bidirectionalGRU(questionWordList)

        # StoryBidirectionalGRU 
        #SotryBidirectionalGRU()
        
        # AnswersBidirectionalGRU 
        #AnswersBidirectionalGRU()
        
        # Attention 
        #Attention()

        return guessAnsList

    def bidirectionalGRU(self, questionWordList):
        # forward vector
        fOneHot = self.oneHotEncoding(questionWordList)
        forwardVector = self.GRU(fOneHot)
        #print(fOneHot.shape)
        # backward vector
        #bOneHot = self.oneHotEncoding(questionWordList.reverse())
        #backwardVector = self.GRU(bOneHot)
        # concat vector
        #concat = tf.concat(0, [forward, backward])

        #print(concat)

    def GRU(self, oneHotEncoding):
        # tf GRU cell
        gru = tf.nn.rnn_cell.GRUCell(num_units=10)
        init_state = cell.zero_state(3, dtype=tf.float32)
        hiddens,states = tf.contrib.rnn.static_rnn(cell=gru_cell,inputs=input_x1,dtype=tf.float32)
        print("done")
        
    def oneHotEncoding(self, WordList):
        # dict transfer to array
        values = array(WordList)
        print(values)
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        print(integer_encoded)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        # create oneHotVector, only need to take maximum number building one-hot vector from array
        oneHotV = np.zeros((len(integer_encoded), integer_encoded.max()+1))
        oneHotV[np.arange(integer_encoded.max()+1), integer_encoded.tolist()] = 1
        print(oneHotV)

        return oneHotV