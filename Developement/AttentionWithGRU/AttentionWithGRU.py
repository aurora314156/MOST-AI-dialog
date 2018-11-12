import os, sys, logging
import tensorflow as tf
import numpy as np
from numpy import array, argmax
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, RepeatVector, TimeDistributed, Input
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
        questionVector = self.bidirectionalGRU(questionWordList)

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
        forwardVector = self.GRUTest(fOneHot)
        print(forwardVector.shape)
        # backward vector
        bOneHot = self.oneHotEncoding(list(reversed(questionWordList)))
        backwardVector = self.GRUTest(bOneHot)
        print(backwardVector.shape)
        # concat forward vector and backward vector
        quesitonVector = np.concatenate((forwardVector,backwardVector), axis=None)
        print(quesitonVector)
        print(quesitonVector.shape)
        print(type(quesitonVector))

        return questionVector

    def GRU(self, oneHotEncoding):
        # keras GRU cell
        # reshape input into [samples, timesteps, features]
        train_x = oneHotEncoding
        print(oneHotEncoding.shape)
        n_in = len(oneHotEncoding)
        train_x = train_x.reshape((1, n_in, 1))
        #parameter
        n_units = 50
        # define model
        model = Sequential()
        model.add(GRU(n_units, activation='relu', input_shape=(n_in,1)))
        model.add(RepeatVector(n_in))
        model.add(GRU(n_units, activation='relu', return_sequences=True, return_state=True))
        model.add(TimeDistributed(Dense(1, activation='relu')))
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(model.summary())
        history = model.fit(train_x, train_x, epochs = 30)
        print()
        
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        print()

    def GRUTest(self, oneHotEncoding):
        # define timesteps
        queVlen = len(oneHotEncoding)
        # define model, save GRU all hidden state and final hidden state for question vector representation
        inputs = Input(shape=(queVlen,1))
        temp_all_hidden_state, temp_final_hidden_state = GRU(50, return_sequences=True, return_state=True, activation='softmax')(inputs)
        model = Model(inputs=inputs, outputs=[temp_all_hidden_state, temp_final_hidden_state])
        # define input data
        data = oneHotEncoding.reshape(1,queVlen,1)
        # train model using encoder method
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        mp = model.predict(data, verbose = 1)
        all_hidden_state, final_hidden_state = mp[0], mp[1]
        
        #print(all_hidden_state)
        #print("================================================")
        #print(final_hidden_state)
        
        return final_hidden_state
        

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


# Plot training & validation accuracy values
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()