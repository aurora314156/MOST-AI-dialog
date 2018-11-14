import os, sys, logging

import tensorflow as tf

import numpy as np

from numpy import array, argmax

from keras.models import Sequential, Model

from keras.layers import LSTM, GRU, Dense, RepeatVector, TimeDistributed, Input

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from numpy import linalg as LA

import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

import gc

sys.path.append('../')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class AttentionWithGRU():

    def __init__(self, CQADataSet, tqn):

        self.CQADataSet = CQADataSet

        self.tqn = tqn



    def AttentionWithGRUMain(self):

        

        guessAnsList = []

        questionWordList = self.CQADataSet[0].getQuestion()

        storyWordList = self.CQADataSet[0].getCorpus()

        # QuestionBidirectionalGRU

        questionVector = self.BidirectionalGRU(questionWordList)

        print("questionVector len:", len(questionVector))

        # StoryBidirectionalGRU 

        storyVector = self.BidirectionalStoryGRU(storyWordList)

        print("storyVector len:", len(storyVector))

        # AttentionValue

        attentionValueVector = self.AttentionValue(storyVector, questionVector)

        print("attentionValueVector length:",len(attentionValueVector))

        print("attentionValueVector sum:", sum(attentionValueVector))

        # WordLevelAttetion

        self.WordLevelAttention(storyVector, attentionValueVector)

        # avoid tensorflow error

        gc.collect()

        return guessAnsList



    def WordLevelAttention(self, storyVector, attentionValueVector):



        storyVector = np.ravel(array(storyVector))

        storyVector = storyVector.tolist()

        wordLevelStoryVector = np.array([(storyVector[i] + storyVector[i+1]) * attentionValueVector[i] for i in range(len(attentionValueVector))])

        print(wordLevelStoryVector.shape)

        print(123)



    def AttentionValue(self, storyVector, questionVector):

        # calculate AttentionValue, using cosine similarity between storyVector and questionVector^2

        attentionValue = []

        for index in range(len(storyVector)):

            attentionValue.append(cosine(storyVector[index], np.square(questionVector)))

        # AttentionValue normalization (actually is softmax in this paper...)

        exps = [np.exp(i) for i in attentionValue]

        sum_of_exps = sum(exps)

        attentionValue_softmax = [j/sum_of_exps for j in exps]



        return attentionValue_softmax



    def BidirectionalGRU(self, questionWordList):

        # forward vector

        fOneHot = self.OneHotEncoding(questionWordList)

        f_all_hidden_state, f_final_hidden_state = self.GRU(fOneHot)

        # backward vector

        bOneHot = self.OneHotEncoding(list(reversed(questionWordList)))

        b_all_hidden_state, b_final_hidden_state = self.GRU(bOneHot)

        # concat forward vector and backward vector

        forwardVector, backwardVector = f_final_hidden_state, b_final_hidden_state

        # print(forwardVector.shape)

        # print(backwardVector.shape)

        quesitonVector = np.concatenate((forwardVector,backwardVector), axis=None)

        # print(quesitonVector)

        # print(quesitonVector.shape)

        # print(type(quesitonVector))

        return quesitonVector



    def BidirectionalStoryGRU(self, storyWordList):

        # forward vector

        fOneHot = self.OneHotEncoding(storyWordList)

        f_all_hidden_state, f_final_hidden_state = self.GRU(fOneHot)

        # print(f_all_hidden_state.shape)

        # backward vector

        bOneHot = self.OneHotEncoding(list(reversed(storyWordList)))

        b_all_hidden_state, b_final_hidden_state = self.GRU(bOneHot)

        # print(b_all_hidden_state.shape)

        # The word vector representation of the t-th word St is constructed 

        # by concatenating the hidden layer outputs of forward and backward GRU networks

        storyVector = []

        for index in range(len(f_all_hidden_state[0])):

           storyVector.append(np.concatenate((f_all_hidden_state[0][index],b_all_hidden_state[0][index]), axis=None))

        

        return storyVector



    def GRU(self, oneHotEncoding):

        # define timesteps

        seqlen = len(oneHotEncoding)

        # define model, save GRU all hidden state and final hidden state for question vector representation

        inputs = Input(shape=(seqlen,1))

        temp_all_hidden_state, temp_final_hidden_state = GRU(1, return_sequences=True, return_state=True, activation='softmax')(inputs)

        model = Model(inputs=inputs, outputs=[temp_all_hidden_state, temp_final_hidden_state])

        # define input data

        data = oneHotEncoding.reshape(1,seqlen,1)

        # train model using encoder method

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        mp = model.predict(data, verbose = 1)

        all_hidden_state, final_hidden_state = mp[0], mp[1]

        

        return all_hidden_state, final_hidden_state



    def GRUModelEvalute(self, oneHotEncoding):

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



    def OneHotEncoding(self, WordList):

        # dict transfer to array

        values = array(WordList)

        # integer encode

        label_encoder = LabelEncoder()

        integer_encoded = label_encoder.fit_transform(values)

        # create one-dim oneHotVector, only need to take maximum number building one-hot vector 

        # from array then merge all one hot encodeing vector to one-dim vector.

        oneHotV = np.zeros((len(integer_encoded), integer_encoded.max()+1))

        oneHotV[np.arange(len(integer_encoded)), integer_encoded] = 1

        oneHotV = oneHotV.ravel()

        

        return oneHotV





# Plot training & validation accuracy values

        # plt.plot(history.history['acc'])

        # plt.plot(history.history['val_acc'])

        # plt.title('Model accuracy')

        # plt.ylabel('Accuracy')

        # plt.xlabel('Epoch')

        # plt.legend(['Train', 'Test'], loc='upper left')

        # plt.show()