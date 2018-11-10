import os, sys, logging
import tensorflow as tf
import numpy as np
from numpy import array, argmax
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
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
        print(fOneHot.shape)
        self.GRU(fOneHot)
        #print(fOneHot.shape)
        # backward vector
        #bOneHot = self.oneHotEncoding(list(reversed(questionWordList)))
        #backwardVector = self.GRU(bOneHot)
        #print(bOneHot.shape)
        # concat vector
        #concat = tf.concat(0, [fOneHot, bOneHot])
        #print(concat)

    def GRU(self, oneHotEncoding):
        # keras GRU cell
        # reshape input into [samples, timesteps, features]
        train_x = oneHotEncoding
        print(oneHotEncoding.shape)
        n_in = len(oneHotEncoding)
        train_x = train_x.reshape((1, n_in, 1))
        # define model
        model = Sequential()
        model.add(GRU(50, activation='relu', input_shape=(n_in,1)))
        model.add(RepeatVector(n_in))
        model.add(GRU(50, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1, activation='relu')))
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(model.summary())
        history = model.fit(train_x, train_x, epochs = 30)
        print()
        # Plot training & validation accuracy values
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        print()

        
    def oneHotEncoding(self, WordList):
        # dict transfer to array
        values = array(WordList)
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        # create one-dim oneHotVector, only need to take maximum number building one-hot vector 
        # from array then merge all one hot encodeing vector to one-dim vector.
        oneHotV = np.zeros((len(integer_encoded), integer_encoded.max()+1))
        oneHotV[np.arange(len(integer_encoded)), integer_encoded] = 1
        oneHotV = oneHotV.ravel()
        #print(oneHotV)

        return oneHotV