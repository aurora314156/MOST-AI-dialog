import os
import sys
import tensorflow as tf
import logging
import numpy as np
from numpy import array
from numpy import argmax
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.models import Model
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
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
        self.GRU(fOneHot)
        
    def GRU(self, oneHotEncoding):
        # tf GRU cell
        batch_size = 4 
        input = tf.random_normal(shape=[3, batch_size, 6], dtype=tf.float32)
        cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state, time_major=True) #time_major如果是True，就表示RNN的steps用第一个维度表示，建议用这个，运行速度快一点。
        #如果是False，那么输入的第二个维度就是steps。
        #如果是True，output的维度是[steps, batch_size, depth]，反之就是[batch_size, max_time, depth]。就是和输入是一样的
        #final_state就是整个LSTM输出的最终的状态，包含c和h。c和h的维度都是[batch_size， n_hidden]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #print(sess.run(output))                                                                                                                                                                                                                                                   
            #print(sess.run(final_state))
            print(sess.run([output,final_state]))

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


    # input1 = Input(shape=(9,9,1), name = "test")
        # lstm1, state_h, state_c = LSTM(128, return_sequences = True, return_state=True)(input1)
        # model= Model(inputs = input1, outputs=[lstm1,state_h, state_c])
        # print("--------------", oneHotEncoding.shape)
        # model.predict(oneHotEncoding[0])
        # print(model.summary())
        # print(lstm1)